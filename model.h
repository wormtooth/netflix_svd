#pragma once

#include "dataset.h"

typedef double dtype;
#define DTYPE_FMT "%lf"

typedef dtype (urow_t)[USERS_NUMBER];
typedef dtype (mrow_t)[MOVIES_NUMBER];

typedef struct {
    /* number of latent features */
    int f;
    
    /* weights for users: F x USERS_NUMBER */
    urow_t *U;
    /* weights for movies: F x MOVIES_NUMBER */
    mrow_t *M;
    
    /* bias or average of all ratings */
    dtype b;
    /* bias for users */
    dtype *bU;
    /* bias for movies */
    dtype *bM;
} model_t;

typedef struct {
    int f;
    urow_t *dU;
    mrow_t *dM;
} model_grad_t;

/* methods for model_t */

extern model_t* model_new(int f);

extern void model_init(model_t *self);

extern void model_free(model_t *self);

extern dtype model_rmse_loss(model_t *self, rating_t *dataset, int sz);

extern int
model_find_bias(model_t *self, rating_t *dataset, int sz, dtype alpha, dtype beta);

extern int
model_compute_grad(model_t *self, rating_t *items, int n, model_grad_t *grad);

extern int
model_update(model_t *self, model_grad_t *grad, dtype lr, dtype gamma);

extern int
model_update_sgd(model_t *self, rating_t *item, dtype lr, dtype gamma);

extern model_t* model_load(const char *path);

extern int model_save(model_t *self, const char *path);

/* methods for model_grad_t */

extern model_grad_t* model_grad_new(int f);

extern void model_grad_free(model_grad_t *self);

extern void model_grad_zero(model_grad_t *self);