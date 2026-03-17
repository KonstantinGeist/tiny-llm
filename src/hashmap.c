#include "hashmap.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

HashMap hm_new(void) {
    HashMap hm = { (HMEntry *)calloc(HM_CAP, sizeof(HMEntry)), HM_CAP };
    return hm;
}

static uint32_t hm_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}

void hm_put(HashMap *hm, const char *key, int val) {
    uint32_t i = hm_hash(key) & (hm->cap-1);
    while (hm->entries[i].key && strcmp(hm->entries[i].key, key) != 0)
        i = (i+1) & (hm->cap-1);
    hm->entries[i].key = key; hm->entries[i].val = val;
}

int hm_get(const HashMap *hm, const char *key) {
    uint32_t i = hm_hash(key) & (hm->cap-1);
    while (hm->entries[i].key) {
        if (strcmp(hm->entries[i].key, key) == 0) return hm->entries[i].val;
        i = (i+1) & (hm->cap-1);
    }
    return -1;
}
