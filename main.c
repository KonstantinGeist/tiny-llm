// Интерактивный CLI для запуска инференса модели Qwen3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/select.h>
#include <sys/time.h>

#include "engine.h"
#include "chat.h"

// Вспомогательные функции терминала

static struct termios saved_termios;

static void term_raw(void) {
    tcgetattr(STDIN_FILENO, &saved_termios);
    struct termios raw = saved_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);
}

static void handle_sigint(int sig) {
    (void)sig;
    term_restore();
    printf("\n");
    exit(0);
}

// Общие данные для главного потока и инференс-потока.

static atomic_int stop_requested = 0;  // если 1, то остановить генерацию

// Буфер ответов, заполняемы в token_cb. Используется для записи ответов ассистента.
typedef struct {
    char *buf;
    int   len;
    int   cap;
} ReplyBuf;

static void reply_append(ReplyBuf *r, const char *piece) {
    int plen = (int)strlen(piece);
    if (r->len + plen + 1 > r->cap) {
        r->cap = r->cap ? r->cap * 2 : 4096;
        if (r->len + plen + 1 > r->cap) r->cap = r->len + plen + 1;
        r->buf = realloc(r->buf, r->cap);
    }
    memcpy(r->buf + r->len, piece, plen);
    r->len += plen;
    r->buf[r->len] = '\0';
}

// Коллбэк тоенов: печатает каждый токен, добавляет его в буфер ответа.
static int token_cb(Engine *e, int token_id, void *ctx) {
    if (atomic_load(&stop_requested)) return 1;
    const char *piece = engine_decode_token(e, token_id);
    fputs(piece, stdout);
    fflush(stdout);
    reply_append((ReplyBuf *)ctx, piece);
    return 0;
}

// Инференс-поток.

typedef struct {
    Engine     *engine;
    const char *prompt;  // сформатированная строка в формате ChatML, время жизни управляется вызывающим кодом
    ReplyBuf   *reply;
} GenArgs;

static pthread_t       s_inf_thread;
static pthread_mutex_t s_job_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  s_job_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  s_job_done  = PTHREAD_COND_INITIALIZER;

static GenArgs *s_job_args    = NULL; // текущая задача, NULL = простой
static int      s_job_pending = 0;    // 1 = ожидается новая задача
static int      s_job_running = 0;    // 1 = задача в прогрессе
static int      s_inf_exit    = 0;    // 1 = нужно завершить генерацию

static void *inf_thread(void *arg) {
    (void)arg;
    pthread_mutex_lock(&s_job_mutex);
    for (;;) {
        // Ожидаем, пока не появится новая задача
        while (!s_job_pending && !s_inf_exit)
            pthread_cond_wait(&s_job_ready, &s_job_mutex);

        if (s_inf_exit) {
            pthread_mutex_unlock(&s_job_mutex);
            return NULL;
        }

        // Забираем задачу
        GenArgs *a    = s_job_args;
        s_job_pending = 0;
        s_job_running = 1;
        pthread_mutex_unlock(&s_job_mutex);

        // Запускаем инференс
        engine_generate(a->engine, a->prompt, token_cb, a->reply);

        pthread_mutex_lock(&s_job_mutex);
        s_job_running = 0;
        pthread_cond_signal(&s_job_done);
    }
}

// Запускаем задачу и ждём завершения, также ждём нажатия Escape для отмены.
static void run_job(GenArgs *a) {
    // Опубликовать задачу
    pthread_mutex_lock(&s_job_mutex);
    s_job_args    = a;
    s_job_pending = 1;
    s_job_running = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);

    // Отслеживаем Escape, пока работает инференс
    for (;;) {
        pthread_mutex_lock(&s_job_mutex);
        int done = !s_job_running;
        pthread_mutex_unlock(&s_job_mutex);
        if (done) break;

        fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = { 0, 20000 }; /* 20 мс */
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
            char ch;
            if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 27) {
                atomic_store(&stop_requested, 1);
                break;
            }
        }
    }

    // Ожидаем завершения инференс-потока
    pthread_mutex_lock(&s_job_mutex);
    while (s_job_running)
        pthread_cond_wait(&s_job_done, &s_job_mutex);
    pthread_mutex_unlock(&s_job_mutex);
}

// Прочитать строку из stdin.
// Возвращает 1, если нажат Escape.
static int read_user_input(char *buf, int maxlen) {
    term_restore();

    fputs(">> ", stdout);
    fflush(stdout);

    int n = 0;
    int c;

    c = fgetc(stdin);
    if (c == 27) {          /* Escape */
        term_raw();
        return 1;
    }
    if (c == EOF || c == '\n') {
        buf[0] = '\0';
        term_raw();
        return 0;
    }
    buf[n++] = (char)c;

    if (fgets(buf + n, maxlen - n, stdin) == NULL) {
        buf[n] = '\0';
    } else {
        int len = (int)strlen(buf);
        if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
    }

    term_raw();
    return 0;
}

// Входная точка.

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int         think_mode = 0; // по умолчанию — без размышлений

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--think") == 0) {
            think_mode = 1;
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "ошибка: неизвестный аргумент '%s'\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "Использование: %s [--think] <model.gguf>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, handle_sigint);

    Engine *e = engine_load(model_path);
    if (!e) { fprintf(stderr, "завершилось с ошибкой.\n"); return 1; }

    fprintf(stderr, "Чат с Qwen3-1.7B-Instruct (нажмите Esc или Ctrl-C для выхода)\n");
    fprintf(stderr, "Режим размышлений: %s\n\n", think_mode ? "включён" : "выключен (--think для включения)");

    pthread_create(&s_inf_thread, NULL, inf_thread, NULL);

    term_raw();

    ChatHistory history;
    chat_init(&history, NULL, think_mode);

    char     user_msg[4096];
    ReplyBuf reply     = { NULL, 0, 0 };
    int      sent_msgs = 0;

    for (;;) {
        int esc = read_user_input(user_msg, sizeof(user_msg));
        if (esc || user_msg[0] == '\0') break;

        printf("\n");
        fflush(stdout);

        chat_append(&history, ROLE_USER, user_msg);
        int close_prev = (sent_msgs > 0);
        char *prompt = chat_format_delta(&history, sent_msgs, close_prev, 1);

        reply.len = 0;
        if (reply.buf) reply.buf[0] = '\0';

        // Запуск задачу на инференс-потоке.
        atomic_store(&stop_requested, 0);
        GenArgs args = { e, prompt, &reply };
        run_job(&args);
        free(prompt);

        // Выйти, если пользователь нажал Escape.
        if (atomic_load(&stop_requested)) break;

        // Записать ответ ассистента в историю чата, для полного контекста.
        if (reply.buf && reply.len > 0) {
            chat_append(&history, ROLE_ASSISTANT, reply.buf);
            sent_msgs = history.len;
        } else {
            sent_msgs = history.len;
        }

        printf("\n\n");
        fflush(stdout);
    }

    // Остановить инференс-поток.
    pthread_mutex_lock(&s_job_mutex);
    s_inf_exit = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);
    pthread_join(s_inf_thread, NULL);

    term_restore();
    printf("\nПока.\n");

    // Статистика скорости.
    EngineStats stats;
    engine_get_stats(e, &stats);
    if (stats.prefill_tokens > 0 || stats.gen_tokens > 0) {
        fprintf(stderr, "\n--- Статистика ---\n");
        if (stats.prefill_tokens > 0)
            fprintf(stderr, "prefill : %ld ткенов  %.0f мс  %.1f ток/сек\n",
                    stats.prefill_tokens, stats.prefill_ms,
                    stats.prefill_tokens / (stats.prefill_ms / 1e3));
        if (stats.gen_tokens > 0)
            fprintf(stderr, "generate: %ld токенов  %.0f мс  %.1f ток/сек\n",
                    stats.gen_tokens, stats.gen_ms,
                    stats.gen_tokens / (stats.gen_ms / 1e3));
    }

    free(reply.buf);
    chat_free(&history);
    engine_free(e);
    return 0;
}
