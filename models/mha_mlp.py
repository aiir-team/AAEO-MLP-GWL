# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# No: TPO
# No: EP, ES, MA
# No: CA, ICA,
# No: SA
# No: BFO, CSO, GOA, BeesA
# No: HC, HS, CEM
# No: Dummy

from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, FPA, GA
from mealpy.human_based import BRO, BSO, CHIO, FBIO, GSKA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CGO, GBO, SCA
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, TWO, WDO
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BES, BSA, COA, CSA, DO, EHO, FA, FFA, FOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, WOA

from config import Config
from models.base_mlp import HybridMlp

## Evolutionary Group

class GaMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.pc = mha_paras["pc"]
        self.pm = mha_paras["pm"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.pc}-{self.pm}"

    def training(self):
        self.optimizer = GA.BaseGA(self.problem, self.epoch, self.pop_size, self.pc, self.pm)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class DeMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.wf = mha_paras["wf"]
        self.cr = mha_paras["cr"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.wf}-{self.cr}"

    def training(self):
        self.optimizer = DE.BaseDE(self.problem, self.epoch, self.pop_size, self.wf, self.cr)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


## Swarm-based group

class PsoMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.c1 = mha_paras["c1"]
        self.c2 = mha_paras["c2"]
        self.w_min = mha_paras["w_min"]
        self.w_max = mha_paras["w_max"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.c1}-{self.c2}-{self.w_min}-{self.w_max}"

    def training(self):
        self.optimizer = PSO.BasePSO(self.problem, self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

class HhoMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = HHO.BaseHHO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class SsaMlp(HybridMlp):
    """
        Sparrow Search Algorithm (SSA): https://github.com/thieu1995/mealpy/blob/master/mealpy/swarm_based/SSA.py
    """

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.ST = mha_paras["ST"]  # ST in [0.5, 1.0], safety threshold value
        self.PD = mha_paras["PD"]  # number of producers (percentage)
        self.SD = mha_paras["SD"]  # number of sparrows who perceive the danger
        self.filename = f"{self.epoch}-{self.pop_size}-{self.ST}-{self.PD}-{self.SD}"

    def training(self):
        self.optimizer = SSA.BaseSSA(self.problem, self.epoch, self.pop_size, self.ST, self.PD, self.SD)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class HgsMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.L = mha_paras["L"]
        self.LH = mha_paras["LH"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.L}-{self.LH}"

    def training(self):
        self.optimizer = HGS.OriginalHGS(self.problem, self.epoch, self.pop_size, self.L, self.LH)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

## Physics-based group

class MvoMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.wep_min = mha_paras["wep_min"]
        self.wep_max = mha_paras["wep_max"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.wep_min}-{self.wep_max}"

    def training(self):
        self.optimizer = MVO.BaseMVO(self.problem, self.epoch, self.pop_size, self.wep_min, self.wep_max)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

class EfoMlp(HybridMlp):
    """Electromagnetic Field Optimization (EFO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.r_rate = mha_paras["r_rate"]  # default = 0.3     # Like mutation parameter in GA but for one variable
        self.ps_rate = mha_paras["ps_rate"]  # default = 0.85    # Like crossover parameter in GA
        self.p_field = mha_paras["p_field"]  # default = 0.1
        self.n_field = mha_paras["n_field"]  # default = 0.45
        self.filename = f"{self.epoch}-{self.pop_size}-{self.r_rate}-{self.ps_rate}-{self.p_field}-{self.n_field}"

    def training(self):
        self.optimizer = EFO.BaseEFO(self.problem, self.epoch, self.pop_size, self.r_rate, self.ps_rate, self.p_field, self.n_field)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

class EoMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = EO.BaseEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


## Human-based group

class ChioMlp(HybridMlp):
    """Coronavirus Herd Immunity Optimization (CHIO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.brr = mha_paras["brr"]  # default = 0.06
        self.max_age = mha_paras["max_age"]  # default = 150
        self.filename = f"{self.epoch}-{self.pop_size}-{self.brr}-{self.max_age}"

    def training(self):
        self.optimizer = CHIO.BaseCHIO(self.problem, self.epoch, self.pop_size, self.brr, self.max_age)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class FbioMlp(HybridMlp):
    """Forensic-Based Investigation Optimization (FBIO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = FBIO.BaseFBIO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


## Bio-based group

class SmaMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.z = mha_paras["z"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.z}"

    def training(self):
        self.optimizer = SMA.BaseSMA(self.problem, self.epoch, self.pop_size, self.z)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


## System-based group

class AeoMlp(HybridMlp):
    """Artificial Ecosystem-based Optimization (AEO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.OriginalAEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class MaeoMlp(HybridMlp):
    """Modified Artificial Ecosystem-based Optimization (AEO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.ModifiedAEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class EaeoMlp(HybridMlp):
    """Enhanced Artificial Ecosystem-based Optimization (AEO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.EnhancedAEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class IaeoMlp(HybridMlp):
    """Improved Artificial Ecosystem-based Optimization (AEO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.IAEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)


class AaeoMlp(HybridMlp):
    """Adaptive Artificial Ecosystem-based Optimization (AEO)"""

    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = AEO.AdaptiveAEO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

## Math-based group


class CgoMlp(HybridMlp):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None):
        super().__init__(base_paras, hybrid_paras)
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def training(self):
        self.optimizer = CGO.OriginalCGO(self.problem, self.epoch, self.pop_size)
        self.solution, self.best_fit = self.optimizer.solve(Config.MHA_MODE_TRAIN_PHASE1)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)
