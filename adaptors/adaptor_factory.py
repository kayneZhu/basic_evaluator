from .base_adaptor import BaseAdaptor
from .aime24_adaptor import AIME24Adaptor
from .aime25_adaptor import AIME25Adaptor
from .math500_adaptor import Math500Adaptor
from .minerva_adaptor import MinervaAdaptor
from .teacher_traces_adaptor import TeacherTracesAdaptor
from .probe100_adaptor import Probe100Adaptor
from .math_numeric_adaptor import MathNumericAdaptor
from .gpqa_diamond_adaptor import GPQADiamondAdaptor
from .scibench_adaptor import SciBenchAdaptor
from .arc_challenge_adaptor import ARCChallengeAdaptor
from .mmlu_pro_adaptor import MMLUProAdaptor
from .verl_aligned_adaptor import VerlAlignedAdaptor
from .verl_gpqa_diamond_adaptor import VerlGPQADiamondAdaptor
from .verl_mmlu_pro_adaptor import VerlMMLUProAdaptor
from .verl_scibench_adaptor import VerlSciBenchAdaptor
from .c1_math_adaptor import (
    C1Aime24Adaptor,
    C1Aime25Adaptor,
    C1Aime26Adaptor,
    C1AimeUnionAdaptor,
    C1Amc23Adaptor,
    C1HeldoutHHardAdaptor,
    C1Hmmt25Adaptor,
    C1Math500Adaptor,
    C1OR1200Adaptor,
)
from .humaneval_plus_adaptor import HumanEvalPlusAdaptor


class AdaptorFactory:
    @staticmethod
    def create_adaptor(benchmark_type: str, data_path: str,
                       thinking_mode: bool = False, **kwargs) -> BaseAdaptor:
        adaptor_map = {
            'aime24': AIME24Adaptor,
            'aime25': AIME25Adaptor,
            'math500': Math500Adaptor,
            'minerva': MinervaAdaptor,
            'teacher_traces_12k': TeacherTracesAdaptor,
            'teacher_traces_new': TeacherTracesAdaptor,
            'teacher-traces': TeacherTracesAdaptor,
            'probe100': Probe100Adaptor,
            'math_numeric': MathNumericAdaptor,
            'math_numeric_3k': MathNumericAdaptor,
            'math_numeric_processed_3k': MathNumericAdaptor,
            'math_numeric_processed_3k_failed_pass4': MathNumericAdaptor,
            'aime24_aime25': TeacherTracesAdaptor,
            'aime26': TeacherTracesAdaptor,
            'aime26_bench_schema': TeacherTracesAdaptor,
            'math500_bench_schema': TeacherTracesAdaptor,
            'gpqa_diamond': GPQADiamondAdaptor,
            'gpqa-diamond': GPQADiamondAdaptor,
            'scibench': SciBenchAdaptor,
            'scibench_train': SciBenchAdaptor,
            'arc_challenge': ARCChallengeAdaptor,
            'arc-challenge': ARCChallengeAdaptor,
            'mmlu_pro': MMLUProAdaptor,
            'mmlu-pro': MMLUProAdaptor,
            'verl_aligned': VerlAlignedAdaptor,
            'verl-aligned': VerlAlignedAdaptor,
            'verl_gpqa_diamond': VerlGPQADiamondAdaptor,
            'verl_mmlu_pro': VerlMMLUProAdaptor,
            'verl_scibench': VerlSciBenchAdaptor,
            # This paper's C.1 mixin (do not alias onto VerlAlignedAdaptor).
            'c1_aime_union': C1AimeUnionAdaptor,
            'aime_union': C1AimeUnionAdaptor,
            'c1_aime24': C1Aime24Adaptor,
            'c1_aime25': C1Aime25Adaptor,
            'c1_aime26': C1Aime26Adaptor,
            'c1_or1_200': C1OR1200Adaptor,
            'or1_200': C1OR1200Adaptor,
            'c1_heldout_h_hard': C1HeldoutHHardAdaptor,
            'c1_h_hard': C1HeldoutHHardAdaptor,
            'c1_math500': C1Math500Adaptor,
            'c1_amc23': C1Amc23Adaptor,
            'c1_hmmt25': C1Hmmt25Adaptor,
            # Stub: raises NotImplementedError on construct. Not a fake scorer.
            'humaneval_plus': HumanEvalPlusAdaptor,
            'humaneval+': HumanEvalPlusAdaptor,
        }

        adaptor_class = adaptor_map.get(benchmark_type.lower())
        if adaptor_class is None:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}. "
                           f"Supported types: {list(adaptor_map.keys())}")

        return adaptor_class(data_path, thinking_mode, **kwargs)
