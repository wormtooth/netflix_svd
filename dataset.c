#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "utils.h"

uint32_t *gUserMappings = NULL;
rating_t *gRatings = NULL;
rating_t *gProbe = NULL;

static int rating_by_uid(const void *a, const void *b);
static int rating_by_mid(const void *a, const void *b);
static int rating_by_uid_mid(const void *a, const void *b);
static int load_quick();
static int load_train();
static int remap_users();
static int load_probe();
static int get_rating(rating_t *item);

static int need_cache = 0;
static int cache();

extern int load_ratings() {
    gUserMappings = safe_malloc(sizeof(uint32_t) * USERS_NUMBER, "gUserMappings");
    gRatings = safe_malloc(sizeof(rating_t) * RATINGS_NUMBER, "gRatings");
    gProbe = safe_malloc(sizeof(rating_t) * PROBE_NUMBER, "gProbe");

    int n = load_train();
    if (n != RATINGS_NUMBER) {
        log_error(
            "Fail to load all %d ratings. %d ratings loaded.",
            RATINGS_NUMBER, n
        );
        exit(1);
    }
    log_debug("Loaded %d ratings.", n);

    n = remap_users();
    if (n != USERS_NUMBER) {
        log_error(
            "Fail to remap all %d users. %d users remapped.",
            USERS_NUMBER, n
        );
        exit(1);
    }
    log_debug("Remaped %d users.", n);

    n = load_probe();
    if (n != PROBE_NUMBER) {
        log_error(
            "Fail to load all %d probe ratings. %d ratings loaded.",
            PROBE_NUMBER, n
        );
        exit(1);
    }
    log_debug("Loaded %d probe ratings.", n);

    cache();
    return 0;
}

extern int unload_ratings() {
    log_debug("Unloading ratings dataset ...");
    
    if (gUserMappings) {
        free(gUserMappings);
        gUserMappings = NULL;
    }

    if (gRatings) {
        free(gRatings);
        gRatings = NULL;
    }

    if (gProbe) {
        free(gProbe);
        gProbe = NULL;
    }

    log_debug("Unloaded ratings dataset.");
    return 0;
}

extern void shuffle_ratings(rating_t *items, int n) {
    rating_t tmp;
    for (int i = n - 1; i > 0; i --) {
        int j = (int) (drand48()*(i+1));
        tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}

extern void sort_ratings_by_uid(rating_t *items, int n) {
    qsort(items, n, sizeof(rating_t), rating_by_uid);
}

extern void sort_ratings_by_mid(rating_t *items, int n) {
    qsort(items, n, sizeof(rating_t), rating_by_mid);
}

extern int mapped_uid(int true_id) {
    if (!gUserMappings) return -1;
    int a = 0, b = USERS_NUMBER - 1;
    while (a <= b) {
        int m = (a + b) / 2;
        if (gUserMappings[m] == true_id) return m;
        if (gUserMappings[m] < true_id) {
            a = m + 1;
        } else {
            b = m - 1;
        }
    }
    return -1;
}

static int rating_by_uid(const void *a, const void *b) {
    rating_t *s = (rating_t *)a;
    rating_t *t = (rating_t *)b;
    if (s->uid < t->uid) {
        return -1;
    } else if (s->uid > t->uid) {
        return 1;
    }
    return 0;
}

static int rating_by_mid(const void *a, const void *b) {
    rating_t *s = (rating_t *)a;
    rating_t *t = (rating_t *)b;
    if (s->mid < t->mid) {
        return -1;
    } else if (s->mid > t->mid) {
        return 1;
    }
    return 0;
}

static int rating_by_uid_mid(const void *a, const void *b) {
    rating_t *s = (rating_t *)a;
    rating_t *t = (rating_t *)b;
    if (s->uid < t->uid) {
        return -1;
    } else if (s->uid > t->uid) {
        return 1;
    }
    if (s->mid < t->mid) {
        return -1;
    } else if (s->mid > t->mid) {
        return 1;
    }
    return 0;
}

static int load_train() {
    FILE *fp;
    char filepath[30];
    char line[100];
    int n;
    int m, u, r;

    fp = fopen("data/ratings.bin", "rb");
    if (fp) {
        log_debug("Cache for ratings is found.");
        n = fread(gRatings, sizeof(rating_t), RATINGS_NUMBER, fp);
        fclose(fp);
        if (n == RATINGS_NUMBER) return n;
        log_error("Cache for ratings is corrupted.");
    }

    n = 0;
    need_cache = 1;
    for (int i = 1; i <= 4; i ++) {
        sprintf(filepath, "data/combined_data_%d.txt", i);
        log_debug("Reading %s ...", filepath);
        fp = fopen(filepath, "r");
        if (!fp) {
            log_error("Cannot open %s", filepath);
            return n;
        }

        while (fgets(line, sizeof(line), fp) != NULL) {
            int l = strlen(line);
            if (line[l - 2] == ':') {
                line[l - 2] = 0;
                m = atoi(line);
                continue;
            }
            sscanf(line, "%d,%d", &u, &r);
            gRatings[n].uid = u;
            gRatings[n].mid = m - 1; // make movid id start from 0
            gRatings[n].rate = r;
            n ++;
        }
        fclose(fp);
        log_debug("Finish reading %s!", filepath);
    }

    return n;
}

static int remap_users() {
    FILE *fp;
    int n, m;

    fp = fopen("data/users.bin", "rb");
    if (fp) {
        log_debug("Cache for users is found.");
        n = fread(gUserMappings, sizeof(uint32_t), USERS_NUMBER, fp);
        fclose(fp);
        if (n == USERS_NUMBER) return n;
        log_error("Cache for users is corrupted.");
    }

    need_cache = 1;
    sort_ratings_by_uid(gRatings, RATINGS_NUMBER);
    n = m = 0;
    while (n < RATINGS_NUMBER) {
        gUserMappings[m] = gRatings[n].uid;
        while (n < RATINGS_NUMBER && gRatings[n].uid == gUserMappings[m]) {
            gRatings[n].uid = m;
            n ++;
        }
        m ++;
    }
    return m;
}

static int load_probe() {
    int n, m, u;
    FILE *fp;
    char line[100];

    fp = fopen("data/probe.bin", "rb");
    if (fp) {
        log_debug("Cache for probe ratings is found.");
        n = fread(gProbe, sizeof(rating_t), PROBE_NUMBER, fp);
        fclose(fp);
        if (n == PROBE_NUMBER) return n;
        log_error("Cache for probe ratings is corrupted.");
    }

    need_cache = 1;
    n = m = 0;
    log_debug("Reading data/probe.txt");
    fp = fopen("data/probe.txt", "r");
    if (!fp) {
        log_error("Cannot open data/probe.txt");
        return n;
    }

    qsort(gRatings, RATINGS_NUMBER, sizeof(rating_t), rating_by_uid_mid);
    while (fgets(line, sizeof(line), fp) != NULL) {
        int l = strlen(line);
        if (line[l - 2] == ':') {
            line[l - 2] = 0;
            m = atoi(line);
            continue;
        }
        sscanf(line, "%d", &u);
        gProbe[n].uid = mapped_uid(u);
        gProbe[n].mid = m - 1; // make movid id start from 0
        gProbe[n].rate = get_rating(&gProbe[n]);
        if (gProbe[n].rate == 0) return n;
        n ++;
    }
    fclose(fp);
    log_debug("Finish reading data/probe.txt!");
    return n;
}

static int get_rating(rating_t *item) {
    int i = 0, j = RATINGS_NUMBER - 1;
    while (i <= j) {
        int k = (i + j) / 2;
        int c = rating_by_uid_mid(item, &gRatings[k]);
        if (c == 0) {
            return gRatings[k].rate;
        } else if (c == -1) {
            j = k - 1;
        } else {
            i = k + 1;
        }
    }
    return 0;
}

static int cache() {
    if (!need_cache) return 0;
    FILE *fp;
    
    fp = fopen("data/ratings.bin", "wb");
    if (!fp) {
        log_error("Cannot open data/ratings.bin");
    } else {
        fwrite(gRatings, sizeof(rating_t), RATINGS_NUMBER, fp);
        fclose(fp);
        log_debug("Cached ratings.");
    }

    fp = fopen("data/users.bin", "wb");
    if (!fp) {
        log_error("Cannot open data/users.bin");
    } else {
        fwrite(gUserMappings, sizeof(uint32_t), USERS_NUMBER, fp);
        fclose(fp);
        log_debug("Cached users.");
    }

    fp = fopen("data/probe.bin", "wb");
    if (!fp) {
        log_error("Cannot open data/probe.bin");
    } else {
        fwrite(gProbe, sizeof(rating_t), PROBE_NUMBER, fp);
        fclose(fp);
        log_debug("Cached probe.");
    }
    
    return 0;
}