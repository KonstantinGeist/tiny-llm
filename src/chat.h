#pragma once

// История диалога.

#include <stddef.h>

typedef enum {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT
} ChatRole;

typedef struct {
    ChatRole  role;
    char     *content;
} ChatMessage;

typedef struct {
    ChatMessage *msgs;
    int          len;
    int          cap;
    int          think;
} ChatHistory;

void chat_init(ChatHistory *h, const char *system_prompt, int think);

void chat_append(ChatHistory *h, ChatRole role, const char *content);

// Форматирует историю диалога в формат ChatML, начинания с сообщения по индексу from_msg и заканчивая h->len-1
// Пользователь должен освободить возвращённую строку через free()
// Флаг add_generation_prompt добавляет "<|im_start|>assistant\n" в конец, чтобы инференс продолжил генерацию.
// Флаг close_prev добавляет "<|im_end|>\n" для завершения предыдущего ответа ассистента.
char *chat_format_delta(const ChatHistory *h, int from_msg,
                        int close_prev, int add_generation_prompt);

void chat_free(ChatHistory *h);
