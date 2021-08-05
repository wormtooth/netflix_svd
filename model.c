#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "utils.h"

dtype *gGradBuffer = NULL;

extern model_t* model_new(int f) {
    model_t *self = safe_malloc(sizeof(model_t), "model_new");
    self->f = f;

    self->U = safe_malloc(sizeof(dtype[f][USERS_NUMBER]), "model_new");
    self->M = safe_malloc(sizeof(dtype[f][MOVIES_NUMBER]), "model_new");
    self->bU = safe_malloc(sizeof(dtype[USERS_NUMBER]), "model_new");
    self->bM = safe_malloc(sizeof(dtype[MOVIES_NUMBER]), "model_new");
    
    return self;
}

extern void model_init(model_t *self) {
    for (int i = 0; i < self->f; i ++) {
        for (int j = 0; j < USERS_NUMBER; j ++) {
            self->U[i][j] = (drand48() * 2 - 1) / sqrt(self->f);
        }
    }

    for (int i = 0; i < self->f; i ++) {
        for (int j = 0; j < MOVIES_NUMBER; j ++) {
            self->M[i][j] = (drand48() * 2 - 1) / sqrt(self->f);
        }
    }
}

extern void model_free(model_t *self) {
    free(self->U);
    free(self->M);
    free(self->bU);
    free(self->bM);
    free(self);
}

extern dtype model_rmse_loss(model_t *self, rating_t *dataset, int sz) {
    dtype loss  = 0.0;
    urow_t *U = self->U;
    mrow_t *M = self->M;
    dtype b = self->b;
    dtype *bU = self->bU;
    dtype *bM = self->bM;
    int f = self->f;
    for (int r = 0; r < sz; r ++) {
        int i = dataset[r].uid;
        int j = dataset[r].mid;
        dtype prod = b + bU[i] + bM[j];
        for (int k = 0; k < f; k ++) {
            prod += U[k][i] * M[k][j];
        }
        prod -= dataset[r].rate;
        loss += prod * prod;
    }
    return sqrt(loss / sz);
}

extern int
model_find_bias(model_t *self, rating_t *dataset, int sz, dtype alpha, dtype beta) {
    dtype s = 0.0;
    for (int i = 0; i < sz; i ++) s += dataset[i].rate;
    self->b = s / sz;

    sort_ratings_by_uid(dataset, sz);
    int p = 0, q = 0;
    int n = 0;
    while (p < sz) {
        s = 0.0;
        q = p;
        while (q < sz && dataset[q].uid == dataset[p].uid) {
            s += (dtype)dataset[q].rate - self->b;
            q ++;
        }
        self->bU[n] = s / (alpha + q - p);
        p = q;
        n ++;
    }

    sort_ratings_by_mid(dataset, sz);
    n = p = q = 0;
    while (p < sz) {
        s = 0.0;
        q = p;
        while (q < sz && dataset[q].mid == dataset[p].mid) {
            s += (dtype)dataset[q].rate - self->b - self->bU[dataset[q].uid];
            q ++;
        }
        self->bM[n] = s / (beta + q - p);
        p = q;
        n ++;
    }
    return 0;
}

extern int
model_compute_grad(model_t *self, rating_t *items, int n, model_grad_t *grad) {
    urow_t *U = self->U, *dU = grad->dU;
    mrow_t *M = self->M, *dM = grad->dM;
    int f = self->f;
    dtype b = self->b;
    dtype *bU = self->bU;
    dtype *bM = self->bM;
    for (int t = 0; t < n; t ++) {
        int i = items[t].uid;
        int j = items[t].mid;
        dtype d = b + bU[i] + bM[j] - (dtype)items[t].rate;
        for (int k = 0; k < f; k ++) {
            d += U[k][i] * M[k][j];
        }
        d /= n;
        for (int k = 0; k < f; k ++) {
            dU[k][i] += 2 * d * M[k][j];
            dM[k][j] += 2 * d * U[k][i];
        }
    }
    return 0;
}

extern int
model_update(model_t *self, model_grad_t *grad, dtype lr, dtype gamma) {
    urow_t *U = self->U, *dU = grad->dU;
    mrow_t *M = self->M, *dM = grad->dM;
    int f = self->f;
    for (int k = 0; k < f; k ++) {
        for (int i = 0; i < USERS_NUMBER; i ++) {
            U[k][i] -= lr * dU[k][i];
        }
        for (int i = 0; i < MOVIES_NUMBER; i ++) {
            M[k][i] -= lr * dM[k][i];
        }
    }

    if (gamma == 0) return 0;
    for (int k = 0; k < f; k ++) {
        for (int i = 0; i < USERS_NUMBER; i ++) {
            U[k][i] -= lr * U[k][i] * gamma * 2;
        }
        for (int i = 0; i < MOVIES_NUMBER; i ++) {
            M[k][i] -= lr * M[k][i] * gamma * 2;
        }
    }
    return 0;
}

extern int
model_update_sgd(model_t *self, rating_t *item, dtype lr, dtype gamma) {
    urow_t *U = self->U;
    mrow_t *M = self->M;
    int f = self->f;
    int i = item->uid;
    int j = item->mid;
    dtype grad[2][self->f];
    dtype d = self->b + self->bU[i] + self->bM[j] - item->rate;
    for (int k = 0; k < f; k ++) {
        d += U[k][i] * M[k][j];
    }
    for (int k = 0; k < f; k ++) {
        grad[0][k] = 2 * d * M[k][j];
        grad[1][k] = 2 * d * U[k][i];
    }
    if (gamma != 0) {
        for (int k = 0; k < f; k ++) {
            grad[0][k] += 2 * gamma * U[k][i];
            grad[1][k] += 2 * gamma * M[k][j];
        }
    }
    for (int k = 0; k < f; k ++) {
        U[k][i] -= lr * grad[0][k];
        M[k][j] -= lr * grad[1][k];
    }
    return 0;
}

extern model_t* model_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    int f;
    model_t *ret = NULL;
    if (!fp) return NULL;
    
    if (fread(&f, sizeof(int), 1, fp) != 1) {
        ret = NULL;
        goto end;
    }
    model_t *self = model_new(f);
    ret = self;
    if (fread(self->U, sizeof(urow_t), f, fp) != f) {
        ret = NULL;
        goto end;
    }
    if (fread(self->M, sizeof(mrow_t), f, fp) != f) {
        ret = NULL;
        goto end;
    }
    if (fread(&self->b, sizeof(dtype), 1, fp) != 1) {
        ret = NULL;
        goto end;
    }
    if (fread(self->bU, sizeof(dtype), USERS_NUMBER, fp) != USERS_NUMBER) {
        ret = NULL;
        goto end;
    }
    if (fread(self->bM, sizeof(dtype), MOVIES_NUMBER, fp) != MOVIES_NUMBER) {
        ret = NULL;
        goto end;
    }

end:
    fclose(fp);
    if (!ret && self) {
        model_free(self);
    }
    return ret;
}

extern int model_save(model_t *self, const char *path) {
    FILE *fp = fopen(path, "wb");
    int ret = 0;
    if (!fp) return -1;

    if (fwrite(&self->f, sizeof(int), 1, fp) != 1) {
        ret = -1;
        goto end;
    }
    if (fwrite(self->U, sizeof(urow_t), self->f, fp) != self->f) {
        ret = -1;
        goto end;
    }
    if (fwrite(self->M, sizeof(mrow_t), self->f, fp) != self->f) {
        ret = -1;
        goto end;
    }
    if (fwrite(&self->b, sizeof(dtype), 1, fp) != 1) {
        ret = -1;
        goto end;
    }
    if (fwrite(self->bU, sizeof(dtype), USERS_NUMBER, fp) != USERS_NUMBER) {
        ret = -1;
        goto end;
    }
    if (fwrite(self->bM, sizeof(dtype), MOVIES_NUMBER, fp) != MOVIES_NUMBER) {
        ret = -1;
        goto end;
    }

end:
    fclose(fp);
    return ret;
}

extern model_grad_t* model_grad_new(int f) {
    model_grad_t *grad = safe_malloc(sizeof(model_grad_t), "model_grad_new");
    grad->f = f;
    grad->dU = safe_malloc(sizeof(dtype[f][USERS_NUMBER]), "model_grad_new");
    grad->dM = safe_malloc(sizeof(dtype[f][MOVIES_NUMBER]), "model_grad_new");
    return grad;
}

extern void model_grad_free(model_grad_t *self) {
    free(self->dU);
    free(self->dM);
    free(self);
}

extern void model_grad_zero(model_grad_t *self) {
    memset(self->dU, 0, sizeof(dtype[self->f][USERS_NUMBER]));
    memset(self->dM, 0, sizeof(dtype[self->f][MOVIES_NUMBER]));
}