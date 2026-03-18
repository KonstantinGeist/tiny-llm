#include "chat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char  *p = malloc(n);
    memcpy(p, s, n);
    return p;
}

static const char *role_name(ChatRole r) {
    switch (r) {
        case ROLE_SYSTEM:    return "system";
        case ROLE_USER:      return "user";
        case ROLE_ASSISTANT: return "assistant";
    }
    return "user";
}

static void push(ChatHistory *h, ChatRole role, const char *content) {
    if (h->len == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 8;
        h->msgs = realloc(h->msgs, h->cap * sizeof(ChatMessage));
    }
    h->msgs[h->len].role    = role;
    h->msgs[h->len].content = xstrdup(content);
    h->len++;
}

void chat_init(ChatHistory *h, const char *system_prompt, int think) {
    h->msgs  = NULL;
    h->len   = 0;
    h->cap   = 0;
    h->think = think;
    if (system_prompt && system_prompt[0])
        push(h, ROLE_SYSTEM, system_prompt);
}

void chat_append(ChatHistory *h, ChatRole role, const char *content) {
    if (role == ROLE_USER && !h->think) {
        // Добавляем /nothink перед сообщением, чтобы отключить режим размышлений.
        size_t n = strlen("/nothink ") + strlen(content) + 1;
        char  *s = malloc(n);
        snprintf(s, n, "/nothink %s", content);
        push(h, role, s);
        free(s);
    } else {
        push(h, role, content);
    }
}

char *chat_format_delta(const ChatHistory *h, int from_msg,
                        int close_prev, int add_generation_prompt) {
    size_t need = 1; // NUL-терминатор
    if (close_prev)
        need += strlen("<|im_end|>\n");
    for (int i = from_msg; i < h->len; i++) {
        need += strlen("<|im_start|>") + strlen(role_name(h->msgs[i].role))
              + 1  /* \n */
              + strlen(h->msgs[i].content)
              + strlen("<|im_end|>")
              + 1; /* \n */
    }
    if (add_generation_prompt)
        need += strlen("<|im_start|>assistant\n");

    char *buf = malloc(need);
    buf[0] = '\0';

    char *p = buf;
    if (close_prev)
        p += sprintf(p, "<|im_end|>\n");
    for (int i = from_msg; i < h->len; i++) {
        p += sprintf(p, "<|im_start|>%s\n%s<|im_end|>\n",
                     role_name(h->msgs[i].role),
                     h->msgs[i].content);
    }
    if (add_generation_prompt)
        p += sprintf(p, "<|im_start|>assistant\n");

    return buf;
}

void chat_free(ChatHistory *h) {
    for (int i = 0; i < h->len; i++)
        free(h->msgs[i].content);
    free(h->msgs);
    h->msgs = NULL;
    h->len  = 0;
    h->cap  = 0;
}
