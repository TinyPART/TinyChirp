#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

#include "sos_filter.h"
#include "resample.h"
#include "min_max_scaler.h"
#include "signal_energy.h"
#include "shell.h"
#include "vfs_default.h"

#include "xtimer.h"
#include "random.h"

#define NUM_OF_SOS 9
#define IMPULSE_SIZE 20
#define SAMPLE_RATE 48000
#define RESAMPLE_RATE 16000
#define STORAGE_DEV "/sd0"
// #define STORAGE_DEV VFS_DEFAULT_NVM(0)

static int _cat(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s <file>\n", argv[0]);
        return 1;
    }
    /* With newlib or picolibc, low-level syscalls are plugged to RIOT vfs
     * on native, open/read/write/close/... are plugged to RIOT vfs */
#if defined(MODULE_NEWLIB) || defined(MODULE_PICOLIBC)
    FILE *f = fopen(argv[1], "r");
    if (f == NULL) {
        printf("file %s does not exist\n", argv[1]);
        return 1;
    }
    char c;
    while (fread(&c, 1, 1, f) != 0) {
        putchar(c);
    }
    fclose(f);
#else
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        printf("file %s does not exist\n", argv[1]);
        return 1;
    }
    char c;
    while (read(fd, &c, 1) != 0) {
        putchar(c);
    }
    close(fd);
#endif
    fflush(stdout);
    return 0;
}

static int _tee(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: %s <file> <str>\n", argv[0]);
        return 1;
    }

#if defined(MODULE_NEWLIB) || defined(MODULE_PICOLIBC)
    FILE *f = fopen(argv[1], "w+");
    if (f == NULL) {
        printf("error while trying to create %s\n", argv[1]);
        return 1;
    }
    if (fwrite(argv[2], 1, strlen(argv[2]), f) != strlen(argv[2])) {
        puts("Error while writing");
    }
    fclose(f);
#else
    int fd = open(argv[1], O_RDWR | O_CREAT, 00777);
    if (fd < 0) {
        printf("error while trying to create %s\n", argv[1]);
        return 1;
    }
    if (write(fd, argv[2], strlen(argv[2])) != (ssize_t)strlen(argv[2])) {
        puts("Error while writing");
    }
    close(fd);
#endif
    return 0;
}

static const shell_command_t shell_commands[] = {
    { "cat", "print the content of a file", _cat },
    { "tee", "write a string in a file", _tee },
    { NULL, NULL, NULL }
};

static float raw_audio_buf[SAMPLE_RATE];
void unit_test(void);
int main(void) {

    unit_test();
    // const char nor_flash_path[] = STORAGE_DEV;
    // struct statvfs statvfs_buf;
    // int err = vfs_statvfs(nor_flash_path, &statvfs_buf);
    // if (err < 0) {
    //   printf("NO FILESYSTEM FOUND at %s, Trying format and mount...\n", nor_flash_path);
    // //   err = vfs_format_by_path(nor_flash_path);
    // //   if (err < 0) {
    // //     printf("VFS FORMAT ERROR: %d, PATH: %s\n", err, nor_flash_path);
    // //   }
    //   err = vfs_mount_by_path(nor_flash_path);
    //   if (err < 0) {
    //     printf("VFS MOUNT ERROR: %d, PATH: %s\n", err, nor_flash_path);
    //   }
    // }
    // if (0 == err) {
    //     printf("VFS MOUNT Successful at: %s\n", nor_flash_path);
    // }
    (void) raw_audio_buf;
    printf("Starting writing dummy audio data...\n");
    for(int i = 0; i < 10; i++) {
        random_bytes(raw_audio_buf, sizeof(raw_audio_buf));

        int fd = open(STORAGE_DEV "/dummy_audio", O_RDWR | O_CREAT);
        if (fd < 0) {
            printf("error while trying to create %s, errno: %d\n", STORAGE_DEV "/dummy_audio", fd);
            return 1;
        }

        uint32_t start, end;
        start =  xtimer_now_usec();
        for(int j = 0; j < 3; j++ ) {
            if (write(fd, raw_audio_buf, sizeof(raw_audio_buf)) != (ssize_t)sizeof(raw_audio_buf)) {
                puts("Error while writing");
            }
        }
        end =  xtimer_now_usec();

        close(fd);
        
        printf("Write Round: %d, usec: %ld\n", i, (long int)(end - start));
    }

    printf("finish writing dummy audio data...\n");



    char line_buf[SHELL_DEFAULT_BUFSIZE];
    shell_run(shell_commands, line_buf, SHELL_DEFAULT_BUFSIZE);
   
    return 0;
}

void unit_test(void) {
        // cutoff = 3932  # Desired cutoff frequency of the filter, Hz
    // fs = RESAMPLE_RATE = 16K # Sampling frequency, Hz
    // order = 18  # Order of the filter
    sos_coeff_t highpass_coeff[NUM_OF_SOS][6] =  { 
        {3.48262720e-05, -6.96525440e-05,  3.48262720e-05,
         1.00000000e+00, -2.67560228e-02,  2.08456815e-03},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -2.71679039e-02,  1.75106169e-02},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -2.80173964e-02,  4.93263821e-02},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -2.93594494e-02,  9.95898520e-02},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -3.12860424e-02,  1.71745912e-01},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -3.39403078e-02,  2.71155244e-01},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -3.75409102e-02,  4.06007429e-01},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -4.24244000e-02,  5.88907176e-01},
       { 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
         1.00000000e+00, -4.91210710e-02,  8.39715402e-01}
        };

    // Initialize the SOS filters
    SOS sos_filters[NUM_OF_SOS];
    for (int i = 0; i < NUM_OF_SOS; i++) {
        init_sos(&sos_filters[i], &highpass_coeff[i][0], &highpass_coeff[i][3]);
    }

    // Define an impulse input signal
    sos_io_t input_signal[IMPULSE_SIZE];
    memset(input_signal, 0, sizeof(input_signal)); 
    input_signal[0] = 1.0;
    int signal_length = sizeof(input_signal) / sizeof(input_signal[0]);
    sos_io_t impulse_response[signal_length];

    // Apply the cascaded SOS filter to the input signal
    apply_cascaded_sos(sos_filters, NUM_OF_SOS, input_signal, impulse_response, signal_length);

    // Print the output signal
    for (int i = 0; i < signal_length; ++i) {
        printf("Output[%d] = %f\n", i, impulse_response[i]);
    }

    const sos_io_t impulse_res_ref[] = {
        3.48262720e-05, -6.16229517e-04,  5.05346763e-03, -2.53184395e-02,
        8.57771062e-02, -2.04420435e-01,  3.41738866e-01, -3.75517352e-01,
        2.01091896e-01,  9.07359061e-02, -2.26703949e-01,  7.12998882e-02,
        1.45155549e-01, -1.18280038e-01, -7.85547369e-02,  1.18782591e-01,
        3.87212725e-02, -1.05379552e-01, -1.75339139e-02,  8.98064136e-02
    };
    sos_io_t sum_of_diff = 0.0;
    for (int i = 0; i < signal_length; ++i) {
        sum_of_diff += fabs(impulse_res_ref[i] - impulse_response[i]);
    }
    printf("Sum of DIFF. IMPULSE: %f \n", sum_of_diff);

    const io_t ori_downsample_inp[] = {
      1,2,3,4,5,6,7,8,9,10,11,12
    };
    const io_t ref_downsample_oup[] = {1,4,7,10};
    io_t downsample_oup[4];
    int downsample_len;
    downsample(ori_downsample_inp, sizeof(ori_downsample_inp) / sizeof(io_t),
              downsample_oup, &downsample_len, SAMPLE_RATE / RESAMPLE_RATE);

    for (int i = 0; i < downsample_len; ++i) {
       printf("Downsample Output[%d]: %f, Ref: %f \n", i, downsample_oup[i], ref_downsample_oup[i]);
    }

    const io_t ref_min_max_oup[] = {0, 1.0/3.0, 6.0/9.0, 1.0};
    min_max_scaler_t scaler;
    init_min_max_scaler(&scaler);
    find_min_max(downsample_oup, downsample_len, &scaler);
    printf("Min: %f, Max: %f \n", scaler.min, scaler.max);
    apply_min_max_scaler(downsample_oup, downsample_len, downsample_oup, &scaler);
    for (int i = 0; i < downsample_len; ++i) {
       printf("Min-Max Scale Output[%d]: %f, Ref: %f \n", i, downsample_oup[i], ref_min_max_oup[i]);
    }

    io_t energy = calculate_energy(downsample_oup, downsample_len);
    const io_t ref_energy = (1.0/3.0) * (1.0/3.0) + (6.0/9.0) * (6.0/9.0) + 1.0;
    printf("Sig. Energy Output: %f, Ref: %f \n", energy, ref_energy);
}