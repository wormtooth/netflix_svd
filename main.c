#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "dataset.h"
#include "model.h"
#include "utils.h"

int main() {
    srand(time(NULL));
    load_ratings();

    // hyper-parameters
    int f = 20; 
    dtype alpha = 25.0, beta = 10.0;
    dtype gamma = 0.001;
    dtype lr = 0.01;

    // train test split
    int test_sz = 1000000;
    int train_sz = RATINGS_NUMBER - test_sz;
    shuffle_ratings(gRatings, RATINGS_NUMBER);
    rating_t *train_set = gRatings;
    rating_t *test_set = gRatings + train_sz;

    model_t *model = model_new(f);
    log_info("Created model with %d latent features.", f);
    model_init(model);
    log_info("Initialized model.");
    model_find_bias(model, train_set, train_sz, alpha, beta);
    log_info(
        "Got bias for the model with alpha = %.2f and beta = %.2f",
        alpha, beta
    );

    log_info("RMSE (train): %lf", model_rmse_loss(model, train_set, train_sz));
    log_info("RMSE (test): %lf", model_rmse_loss(model, test_set, test_sz));
    for (int t = 0; t < 5; t ++) {
        shuffle_ratings(train_set, train_sz);
        log_info("Shuffled ratings.");
        for (int i = 0; i < train_sz; i ++) {
            model_update_sgd(model, &(train_set[i]), lr, gamma);
        }
        log_info("Epoch %d done.", t + 1);
        log_info("RMSE (train): %lf", model_rmse_loss(model, train_set, train_sz));
        log_info("RMSE (test): %lf", model_rmse_loss(model, test_set, test_sz));
    }

    model_save(model, "data/model.bin");
    model_free(model);
    unload_ratings();
    return 0;
}