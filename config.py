# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from os.path import abspath, dirname

basedir = abspath(dirname(__file__))


VALIDATION_USED = False

if VALIDATION_USED:
    FILE_METRIC_CSV = ["MAE_train", "RMSE_train", "R_train", "R2_train", "MAPE_train", "NNSE_train",
                       "KGE_train", "PCD_train", "KLD_train", "VAF_train", "A30_train", "A20_train",
                       "MAE_valid", "RMSE_valid", "R_valid", "R2_valid", "MAPE_valid", "NNSE_valid",
                       "KGE_valid", "PCD_valid", "KLD_valid", "VAF_valid", "A30_valid", "A20_valid",
                       "MAE_test", "RMSE_test", "R_test", "R2_test", "MAPE_test", "NNSE_test",
                       "KGE_test", "PCD_test", "KLD_test", "VAF_test", "A30_test", "A20_test"]
    FILE_LOSS_HEADER = ["epoch", "loss", "val_loss"]
else:
    FILE_METRIC_CSV = ["MAE_train", "RMSE_train", "R_train", "R2_train", "MAPE_train", "NNSE_train",
                       "KGE_train", "PCD_train", "KLD_train", "VAF_train", "A30_train", "A20_train",
                       "MAE_test", "RMSE_test", "R_test", "R2_test", "MAPE_test", "NNSE_test",
                       "KGE_test", "PCD_test", "KLD_test", "VAF_test", "A30_test", "A20_test"]
    FILE_LOSS_HEADER = ["epoch", "loss"]


class Config:
    DATA_DIRECTORY = f'{basedir}/data'
    DATA_INPUT = f'{DATA_DIRECTORY}/input_data'
    DATA_RESULTS = f'{DATA_DIRECTORY}/results_paper'
    RESULTS_FOLDER_VISUALIZE = "visualize"

    FILENAME_LOSS_TRAIN = "loss_train"
    FILENAME_PRED_TRAIN = "pred_train"
    FILENAME_PRED_VALID = "pred_valid"
    FILENAME_PRED_TEST = "pred_test"
    FILENAME_PRED_REAL_WORLD = "pred_real_world"
    FILENAME_VISUAL_CONVERGENCE = "convergence"
    FILENAME_VISUAL_PERFORMANCE = "performance"

    FILENAME_METRICS = "metrics"
    FOLDERNAME_STATISTICS = "statistics"
    FILE_MIN = "min.csv"
    FILE_MEAN = "mean.csv"
    FILE_MAX = "max.csv"
    FILE_STD = "std.csv"
    FILE_CV = "cv.csv"
    FILENAME_STATISTICS_FINAL = "statistics_final"

    FILE_METRIC_CSV_HEADER = ["model_paras", "time_train", "time_total",] + FILE_METRIC_CSV
    FILE_METRIC_CSV_HEADER_FULL = ["network", "obj", "trial", "model", "model_paras", "time_train", "time_total",] + FILE_METRIC_CSV
    FILE_METRIC_CSV_HEADER_CALCULATE = ["time_train", "time_total",] + FILE_METRIC_CSV
    FILE_METRIC_HEADER_STATISTICS = ["network", "obj", "model", "model_paras", "time_train", "time_total",] + FILE_METRIC_CSV
    FILE_LOSS_HEADER = FILE_LOSS_HEADER

    FILE_PRED_HEADER = ["Y_test_true_unscaled", "Y_test_pred_unscaled"]
    FILE_METRICS_SAVE_TO_FIGURE = ["R2_test", "A20_test"]

    FILE_FIGURE_TYPES = [".png", ".pdf"]

    LEGEND_NETWORK = "Network = "
    LEGEND_EPOCH = "Number of Generations = "
    LEGEND_POP_SIZE = "Population size = "

    LEGEND_GROUNDTRUTH = "Ground Truth"
    LEGEND_PREDICTED = "Predicted"

    Y_TRAIN_TRUE_SCALED = "y_train_true_scaled"
    Y_TRAIN_TRUE_UNSCALED = "y_train_true_unscaled"
    Y_TRAIN_PRED_SCALED = "y_train_pred_scaled"
    Y_TRAIN_PRED_UNSCALED = "y_train_pred_unscaled"
    Y_VALID_TRUE_SCALED = "y_valid_true_scaled"
    Y_VALID_TRUE_UNSCALED = "y_valid_true_unscaled"
    Y_VALID_PRED_SCALED = "y_valid_pred_scaled"
    Y_VALID_PRED_UNSCALED = "y_valid_pred_unscaled"
    Y_TEST_TRUE_SCALED = "y_test_true_scaled"
    Y_TEST_TRUE_UNSCALED = "y_test_true_unscaled"
    Y_TEST_PRED_SCALED = "y_test_pred_scaled"
    Y_TEST_PRED_UNSCALED = "y_test_pred_unscaled"

    HEADER_TRUTH_PREDICTED_TRAIN_FILE = ["y_train_true_scaled", "y_train_pred_scaled", "y_train_true_unscaled", "y_train_pred_unscaled"]
    HEADER_TRUTH_PREDICTED_VALID_FILE = ["y_valid_true_scaled", "y_valid_pred_scaled", "y_valid_true_unscaled", "y_valid_pred_unscaled"]
    HEADER_TRUTH_PREDICTED_TEST_FILE = ["y_test_true_scaled", "y_test_pred_scaled", "y_test_true_unscaled", "y_test_pred_unscaled"]
    MHA_MODE_TRAIN_PHASE1 = "sequential"        # Don't change this value

    SEED = 20
    #SCALING = "boxcox"  # std, minmax, loge, kurtosis, kurtosis_std, kurtosis_minmax, boxcox, boxcox_minmax, boxcox_std
    VERBOSE_LATEST = "None"
    VERBOSE_SIMPLE_MLP = 2      # 0: Do nothing, 1: print out everything, 2: print training epoch
    N_TRIALS = 10                # Number of trials for each model

    MHA_LB = [-1]  # Lower bound for metaheuristics
    MHA_UB = [1]  # Upper bound for metaheuristics
    ## Training. For Simple MLP, currently support: "RMSE", "MAE", "MSE", "ME"
    OBJ_FUNCS = ["MSE"]     # Metric for training phase in network
    METRICS_FOR_TESTING_PHASE = ["MAE", "RMSE", "R", "R2", "MAPE", "NNSE", "KGE", "PCD", "KLD", "VAF", "A30", "A20"]
    DATA = [
        {
            "name_data": "Ganjimutt_MIR_05", "name_folder": f"{DATA_INPUT}/Ganjimutt_MIR", "name_file": "Ganjimutt_well_MI_0.5.csv",
            "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_4", "GWL_t_5", "GWL_t_6", "GWL_t_7", "GWL_t_9",
                           "GWL_t_10", "GWL_t_11", "GWL_t_12", "P_t_0", "P_t_1"],
            "name_output": "GWL_t_output",
            "scaling": "std", "train_size": 0.7, "valid_size": 0.3, "validation_used": VALIDATION_USED
        },
        # {
        #     "name_data": "Surathkal_MIR_04", "name_folder": f"{DATA_INPUT}/Surathkal_MIR", "name_file": "Surathkal_well_inputs_MI_0.4.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_7", "GWL_t_8", "GWL_t_12", "P_t_0", "P_t_6", "P_t_9", "P_t_12", "T_t_3"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "std", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },

        # Trial Kurtosis Minmax SCaling
        # {
        #     "name_data": "Ganjimutt_MIR_05", "name_folder": f"{DATA_INPUT}/Ganjimutt_MIR", "name_file": "Ganjimutt_well_MI_0.5.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_4", "GWL_t_5", "GWL_t_6", "GWL_t_7", "GWL_t_9",
        #                    "GWL_t_10", "GWL_t_11", "GWL_t_12", "P_t_0", "P_t_1"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "kurtosis_minmax", "train_size": 0.7, "valid_size": 0.3, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_MIR_04", "name_folder": f"{DATA_INPUT}/Surathkal_MIR", "name_file": "Surathkal_well_inputs_MI_0.4.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_7", "GWL_t_8", "GWL_t_12", "P_t_0", "P_t_6", "P_t_9", "P_t_12", "T_t_3"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "kurtosis_minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },


        # {
        #     "name_data": "Ganjimutt_MIR_04", "name_folder": f"{DATA_INPUT}/Ganjimutt_MIR", "name_file": "Ganjimutt_well_MI_0.4.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_4", "GWL_t_5", "GWL_t_6", "GWL_t_7", "GWL_t_8", "GWL_t_9",
        #                    "GWL_t_10", "GWL_t_11", "GWL_t_12", "P_t_0", "P_t_1", "P_t_2", "P_t_7", "P_t_9", "P_t_12"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.3, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Ganjimutt_MIR_05", "name_folder": f"{DATA_INPUT}/Ganjimutt_MIR", "name_file": "Ganjimutt_well_MI_0.5.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_4", "GWL_t_5", "GWL_t_6", "GWL_t_7", "GWL_t_9",
        #                    "GWL_t_10", "GWL_t_11", "GWL_t_12", "P_t_0", "P_t_1"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.3, "validation_used": VALIDATION_USED
        # },

        # {
        #     "name_data": "Surathkal_well_MIR_04", "name_folder": f"{DATA_INPUT}/Surathkal_MIR", "name_file": "Surathkal_well_inputs_MI_0.4.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_7", "GWL_t_8", "GWL_t_12", "P_t_0", "P_t_6", "P_t_9", "P_t_12", "T_t_3"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_well_MIR_045", "name_folder": f"{DATA_INPUT}/Surathkal_MIR", "name_file": "Surathkal_well_inputs_MI_0.45.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_12", "P_t_0"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # }


        # {
        #     "name_data": "Surathkal_well_MIR", "name_folder": f"{DATA_INPUT}/Surathkal_well_MIR_inputs", "name_file": "Surathkal_well_inputs.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_3", "GWL_t_7", "GWL_t_8", "GWL_t_12",
        #                    "P_t_0", "P_t_6", "P_t_9", "P_t_12",
        #                     "T_t_3"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # }

        # {
        #     "name_data": "Surathkal_well", "name_folder": f"{DATA_INPUT}/Surathkal_well", "name_file": "Surathkal_inputs.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_6", "GWL_t_7", "GWL_t_8", "GWL_t_9",
        #                    "P_t_0", "P_t_1", "P_t_3", "P_t_4", "P_t_5", "P_t_6", "P_t_7", "P_t_10",
        #                     "T_t_0","T_t_1", "T_t_2","T_t_5","T_t_6", "T_t_8","T_t_9","T_t_10",
        #                     "TH_t_0", "TH_t_1", "TH_t_3","TH_t_4", "TH_t_5", "TH_t_6","TH_t_7","TH_t_10"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # }

        # {
        #     "name_data": "Surathkal_well_no_TH", "name_folder": f"{DATA_INPUT}/Surathkal_well_important_lags", "name_file": "Surathkal_inputs_impt_no_TH.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "GWL_t_9",
        #                    "P_t_0", "P_t_1",  "P_t_4", "P_t_5",
        #                     "T_t_0","T_t_1",  "T_t_8","T_t_9","T_t_10",
        #                     ],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },

        # {
        #     "name_data": "Surathkal_GWL_P_lag_2", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_lag_2.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "P_t_0",
        #                    "P_t_1", "P_t_2"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # }
        # {
        #     "name_data": "Surathkal_GWL_P_lag_2_no_lag_0", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_lag_2_no_lag_0.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "P_t_1", "P_t_2"
        #                    ],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_GWL_P_T_lag_1", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_T_lag_1.csv",
        #     "name_input": ["GWL_t_1",  "P_t_0",
        #                    "P_t_1", "T_t_0", "T_t_1",
        #                   ],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_GWL_P_T_lag_1_no_lag_0", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_T_lag_1_no_lag_0.csv",
        #     "name_input": ["GWL_t_1",
        #                    "P_t_0", "P_t_1",
        #                    "T_t_0", "T_t_1"
        #                    ],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_GWL_P_T_lag_2", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_T_lag_2.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2", "P_t_0",
        #                    "P_t_1", "P_t_2", "T_t_0", "T_t_1",
        #                    "T_t_2"],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # },
        # {
        #     "name_data": "Surathkal_GWL_P_T_lag_2_no_lag_0", "name_folder": f"{DATA_INPUT}/Surathkal_latest_13_March",
        #     "name_file": "Surathkal_GWL_P_T_lag_2_no_lag_0.csv",
        #     "name_input": ["GWL_t_1", "GWL_t_2","P_t_1", "P_t_2",
        #                    "T_t_1", "T_t_2"
        #                    ],
        #     "name_output": "GWL_t_output",
        #     "scaling": "minmax", "train_size": 0.7, "valid_size": 0.2, "validation_used": VALIDATION_USED
        # }
    ]

    LIST_NETWORKS = [
        [
            {"n_nodes": 5, "activation": "elu", "dropout": 0.2},  # First hidden layer
            {"n_nodes": 1, "activation": "elu"}  # Output layer
        ],
        # [
        #     {"n_nodes": 6, "activation": "relu", "dropout": 0.2},  # First hidden layer
        #     {"n_nodes": 1, "activation": "elu"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "relu", "dropout": 0},  # First hidden layer
        #     {"n_nodes": 1, "activation": "relu"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "elu", "dropout": 0},  # First hidden layer
        #     {"n_nodes": 1, "activation": "elu"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "elu", "dropout": 0},  # First hidden layer
        #     {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "sigmoid", "dropout": 0.2},  # First hidden layer
        #     {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "tanh", "dropout": 0.2},  # First hidden layer
        #     {"n_nodes": 1, "activation": "sigmoid"}  # Output layer
        # ],
        # [
        #     {"n_nodes": 4, "activation": "tanh", "dropout": 0.2},  # First hidden layer
        #     {"n_nodes": 1, "activation": "elu"}  # Output layer
        # ],

        # [
        #     {"n_nodes": 10, "activation": "relu", "dropout": 0.2},      # First hidden layer
        #     {"n_nodes": 5, "activation": "relu", "dropout": 0.2},      # Second hidden layer
        #     {"n_nodes": 1, "activation": "sigmoid"}                    # Output layer
        # ],
    ]

    ## MAE_train,RMSE_train,R_train,R2_train,MAPE_train,NNSE_train,KGE_train,PCD_train,KLD_train,VAF_train,A30_train,A20_train
    ## MAE_test,RMSE_test,R_test,R2_test,MAPE_test,NNSE_test,KGE_test,PCD_test,KLD_test,VAF_test,A30_test,A20_test
    PHASE1_BEST_METRICS = ["RMSE_train", "MAPE_train", "RMSE_test", "MAPE_test"]
    PHASE1_BEST_WEIGHTS = [0.1, 0.1, 0.6, 0.2]

    ## Training model with train and validation dataset. Fitness = train_metric * x1 + valid_metric * x2
    TRAIN_TEST_OBJ_WEIGHTS_FOR_METRICS = [0.0, 1.0]     # [x1, x2]


class MhaConfig:
    EPOCH = [1000]         # Number of generations or epoch in neural network and metaheuristics
    POP_SIZE = [50]     # Number of population size in metaheuristics

    mlp = {
        "epoch": EPOCH,
        "optimizer": ["adam",],    # https://keras.io/api/optimizers/         "RMSprop", "Adadelta", "SGD",
        "learning_rate": [0.02],
        "batch_size": [64],
    }

    ## Evolutionary-based group
    ga_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pc": [0.85],  # crossover probability
        "pm": [0.05]  # mutation probability
    }
    de_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wf": [0.85],  # weighting factor
        "cr": [0.8],  # crossover rate
    }

    ## Swarm-based group
    pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [1.2],  # local coefficient
        "c2": [1.2],  # global coefficient
        "w_min": [0.4],  # weight min factor
        "w_max": [0.9],  # weight max factor
    }
    hho_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ssa_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "ST": [0.8],  # ST in [0.5, 1.0], safety threshold value
        "PD": [0.2],  # number of producers
        "SD": [0.1],  # number of sparrows who perceive the danger
    }
    hgs_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "L": [0.03],  # Switching updating  position probability
        "LH": [1000],  # Largest hunger / threshold
    }

    ## Physics-based group
    mvo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wep_min": [0.2],  # Wormhole Existence Probability (min in Eq.(3.3) paper, default = 0.2
        "wep_max": [1.0],  # Wormhole Existence Probability (max in Eq.(3.3) paper, default = 1.0
    }
    efo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "r_rate": [0.3],  # default = 0.3     # Like mutation parameter in GA but for one variable, mutation probability
        "ps_rate": [0.85],  # default = 0.85    # Like crossover parameter in GA, crossover probability
        "p_field": [0.1],  # default = 0.1     # portion of population, positive field
        "n_field": [0.45],  # default = 0.45    # portion of population, negative field
    }
    eo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Human-based group
    chio_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "brr": [0.06, ],
        "max_age": [150, ],  # maximum number of age
    }
    fbio_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Bio-based group
    sma_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "z": [0.03],  # probability threshold
    }

    ## System-based group
    aeo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    ## Math-based group
    cgo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }

    models = [
        # {"name": "MLP", "class": "MLP", "param_grid": mlp}       # get statistics for MLP models
        #
        #Evolutionary-based
        {"name": "GA-MLP", "class": "GaMlp", "param_grid": ga_paras},  # Genetic Algorithm (GA)
        {"name": "DE-MLP", "class": "DeMlp", "param_grid": de_paras},  # Differential Evolution (DE)

        ## Swarm-based
        {"name": "PSO-MLP", "class": "PsoMlp", "param_grid": pso_paras},  # Particle Swarm Optimization (PSO)
        {"name": "HHO-MLP", "class": "HhoMlp", "param_grid": hho_paras},  # Harris Hawks Optimization (HHO)
        {"name": "SSA-MLP", "class": "SsaMlp", "param_grid": ssa_paras},  # Sparrow Search Algorithm (SSA)
        {"name": "HGS-MLP", "class": "HgsMlp", "param_grid": hgs_paras},  # Hunger Games Search (HGS)

        # ## Physics-based
        {"name": "MVO-MLP", "class": "MvoMlp", "param_grid": mvo_paras},  # Multi-Verse Optimizer (MVO)
        {"name": "EFO-MLP", "class": "EfoMlp", "param_grid": efo_paras},  # Electromagnetic Field Optimization (EFO)
        {"name": "EO-MLP", "class": "EoMlp", "param_grid": eo_paras},  # Equilibrium Optimizer (EO)

        ## Human-based added
        {"name": "CHIO-MLP", "class": "ChioMlp", "param_grid": chio_paras},  # Coronavirus Herd Immunity Optimization (CHIO)
        {"name": "FBIO-MLP", "class": "FbioMlp", "param_grid": fbio_paras},  # Forensic-Based Investigation Optimization (FBIO)

        ## Bio-based
        {"name": "SMA-MLP", "class": "SmaMlp", "param_grid": sma_paras},  # Slime Mould Algorithm (SMA)

        # # Math-based
        {"name": "CGO-MLP", "class": "CgoMlp", "param_grid": cgo_paras},  # Chaos Game Optimization (CGO)

        ## System-based
        {"name": "AEO-MLP", "class": "AeoMlp", "param_grid": aeo_paras},  # Artificial Ecosystem-based Optimization (AEO)
        {"name": "IAEO-MLP", "class": "IaeoMlp", "param_grid": aeo_paras},
        {"name": "EAEO-MLP", "class": "EaeoMlp", "param_grid": aeo_paras},
        {"name": "MAEO-MLP", "class": "MaeoMlp", "param_grid": aeo_paras},
        {"name": "AAEO-MLP", "class": "AaeoMlp", "param_grid": aeo_paras}
    ]
