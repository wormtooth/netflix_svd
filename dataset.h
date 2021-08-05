#pragma once

#include <stdint.h>

/**
 * @brief Information for each rating
 * 
 * uid: user id, remapped to 0 .. 480188, mapping stored in gUserMappings
 * mid: movie id, remapped to 0 .. 17769, true_id = mid + 1
 * rate: the rating of the movie gave by the user, 1 .. 5
 * 
 */
typedef struct {
    uint32_t uid;
    uint16_t mid;
    uint8_t rate;
} rating_t;

#define RATINGS_NUMBER 100480507
#define MOVIES_NUMBER 17770
#define USERS_NUMBER 480189
#define PROBE_NUMBER 1408395

extern uint32_t *gUserMappings;
extern rating_t *gRatings;
extern rating_t *gProbe;

extern int load_ratings();

extern int unload_ratings();

extern void shuffle_ratings(rating_t *items, int n);

extern void sort_ratings_by_uid(rating_t *items, int n);

extern void sort_ratings_by_mid(rating_t *items, int n);

extern int mapped_uid(int true_id);