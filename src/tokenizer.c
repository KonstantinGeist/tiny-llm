#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Специальные токены, длинные впереди.
static const char *SPECIAL[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", NULL
};

void tok_init(Tokenizer *t,
              char **vocab, int vocab_size,
              char **merges, int n_merges,
              int bos_id, int eos_id) {
    t->vocab      = vocab;
    t->vocab_size = vocab_size;
    t->bos_id     = bos_id;
    t->eos_id     = eos_id;

    t->vocab_map = hm_new();
    for (int i = 0; i < vocab_size; i++)
        hm_put(&t->vocab_map, vocab[i], i);

    t->merge_map = hm_new();
    for (int i = 0; i < n_merges; i++)
        hm_put(&t->merge_map, merges[i], i);
}

int tok_encode(const Tokenizer *t, const char *text, int *out) {
    // Предобработка: заменить голые байты их аналогами  в GPT.
    // В данный момент поддерживается только \n → Ċ для ChatML.
    size_t raw_len = strlen(text);
    char *norm = malloc(raw_len * 2 + 1);
    int nlen = 0;
    for (size_t i = 0; i < raw_len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n') {
            norm[nlen++] = (char)0xC4;
            norm[nlen++] = (char)0x8A;  /* Ċ */
        } else {
            norm[nlen++] = (char)c;
        }
    }
    norm[nlen] = '\0';

    int n = 0, len = nlen, pos = 0;
    char buf[256];

    /* Greedy longest-match initial segmentation */
    // Первоначальная сегментация, жадное сопоставление с наиболее длинным токеном.
    while (pos < len) {
        // Сначала специальные токены
        int special_match = -1, special_len = 0;
        for (int s = 0; SPECIAL[s]; s++) {
            int sl = (int)strlen(SPECIAL[s]);
            if (pos + sl <= len && memcmp(norm + pos, SPECIAL[s], sl) == 0) {
                int id = hm_get(&t->vocab_map, SPECIAL[s]);
                if (id >= 0) { special_match = id; special_len = sl; break; }
            }
        }
        if (special_match >= 0) {
            out[n++] = special_match;
            pos += special_len;
            continue;
        }

        // Жадное сопоставление по самому длинному токену
        int max_len = len - pos;
        for (int s = 0; SPECIAL[s]; s++) {
            int sl = (int)strlen(SPECIAL[s]);
            for (int off = 1; off < max_len; off++) {
                if (off + sl <= len - pos &&
                        memcmp(norm + pos + off, SPECIAL[s], sl) == 0) {
                    if (off < max_len) max_len = off;
                    break;
                }
            }
        }
        int try = max_len < 64 ? max_len : 64;
        int match = -1, mlen = 0;
        for (int l = try; l >= 1; l--) {
            memcpy(buf, norm + pos, l); buf[l] = '\0';
            int id = hm_get(&t->vocab_map, buf);
            if (id >= 0) { match = id; mlen = l; break; }
        }
        if (match < 0) { pos++; continue; }
        out[n++] = match;
        pos += mlen;
    }

    free(norm);

    // Слияния BPE: последовательно применяем слияние соседних пар с наименьшим рангом
    char pair[256];
    int changed = 1;
    while (changed) {
        changed = 0;
        int best_rank = INT32_MAX, best_i = -1, prev = -1;
        for (int i = 0; i < n; i++) {
            if (out[i] < 0) continue;
            if (prev >= 0) {
                snprintf(pair, sizeof(pair), "%s %s",
                         t->vocab[prev], t->vocab[out[i]]);
                int rank = hm_get(&t->merge_map, pair);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank; best_i = i;
                }
            }
            prev = out[i];
        }
        if (best_i < 0) break;
        int left = best_i - 1;
        while (left >= 0 && out[left] < 0) left--;
        snprintf(pair, sizeof(pair), "%s%s",
                 t->vocab[out[left]], t->vocab[out[best_i]]);
        int merged = hm_get(&t->vocab_map, pair);
        if (merged < 0) break;
        out[left] = merged; out[best_i] = -1;
        changed = 1;
    }

    int j = 0;
    for (int i = 0; i < n; i++) if (out[i] >= 0) out[j++] = out[i];
    return j;
}

const char *tok_decode(const Tokenizer *t, int token_id) {
    static char buf[512];
    const char *piece = t->vocab[token_id];
    // Конвертация GPT-символов в стандартные:
    //   Ġ (0xC4 0xA0) → ' '
    //   Ċ (0xC4 0x8A) → '\n'
    int has_escape = 0;
    for (const char *p = piece; *p; p++) {
        if ((unsigned char)p[0] == 0xC4 &&
                ((unsigned char)p[1] == 0xA0 || (unsigned char)p[1] == 0x8A)) {
            has_escape = 1; break;
        }
    }
    if (!has_escape) return piece;

    int out = 0;
    for (const char *p = piece; *p && out < (int)sizeof(buf) - 1; ) {
        if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0xA0) {
            buf[out++] = ' '; p += 2;
        } else if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0x8A) {
            buf[out++] = '\n'; p += 2;
        } else {
            buf[out++] = *p++;
        }
    }
    buf[out] = '\0';
    return buf;
}

void tok_free(Tokenizer *t) {
    free(t->vocab_map.entries);
    free(t->merge_map.entries);
}
