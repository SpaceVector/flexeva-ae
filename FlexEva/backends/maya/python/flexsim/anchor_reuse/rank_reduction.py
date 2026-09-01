"""Explicit rank-level reduction substrate for anchor generation and grounding.

This module makes the SPMD -> SPSD / representative-rank policy explicit:

- target world size is the intended full parallel execution width,
- execute world size is the concrete width we actually run for a grounding step,
- representative rank groups describe which ranks stand in for which subset,
- and the plan records whether the reduction preserves portable semantics or
  requires target-cluster realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dryrun_bridge import BoundaryContract
    from .dryrun_bridge import (
        BoundaryContextCapsule,
        EmissionDAG,
        EmissionOwnershipMap,
        LogicPointOwnershipMap,
        LogicSliceGraph,
        LogicStateStore,
    )
    from .schema import Obligation
    from .program_logic import ProgramLogicPoint
    from .semantic_basis import RuntimeValuePoint


class RankReductionKind(str, Enum):
    SEMANTIC_SUBSET = "semantic_subset"
    CLUSTER_CALIBRATION_SUBSET = "cluster_calibration_subset"
    GROUP_REPLICATED_SUBSET = "group_replicated_subset"
    REPRESENTATIVE_FULL_WORLD = "representative_full_world"


class RankReductionCoverageKind(str, Enum):
    EXECUTION_DOMAIN_ONLY = "execution_domain_only"
    REPLICATED_GROUP_COVERAGE = "replicated_group_coverage"
    TARGET_WORLD_COVERAGE = "target_world_coverage"


class RankReductionRequirementKind(str, Enum):
    PORTABLE_PROGRAM_LOGIC = "portable_program_logic"
    RANK_PARTITION_BEHAVIOR = "rank_partition_behavior"
    TARGET_CLUSTER_REALIZATION = "target_cluster_realization"
    REPLICATED_GROUP_COVERAGE = "replicated_group_coverage"
    FULL_TARGET_WORLD_COVERAGE = "full_target_world_coverage"
    EXACT_PER_RANK_REALIZATION = "exact_per_rank_realization"


class RankReductionEvidenceKind(str, Enum):
    POLICY_DEFAULT = "policy_default"
    R_VARIANT_WORLD_UNIFORM = "r_variant_world_uniform"
    R_VARIANT_RANK_PARTITIONING = "r_variant_rank_partitioning"
    OPEN_OBLIGATION_CLUSTER_REALIZATION = "open_obligation_cluster_realization"
    BOUNDARY_CAPSULE_REQUIREMENTS = "boundary_capsule_requirements"
    HOMOGENEOUS_GROUP_LIFT = "homogeneous_group_lift"
    PROMOTED_HIGH_DRIFT = "promoted_high_drift"


class RankReductionEscalationKind(str, Enum):
    SATISFIED = "satisfied"
    STRENGTHEN_REDUCTION = "strengthen_reduction"
    FULL_SPSD_DRY_RUN = "full_spsd_dry_run"


class GroundingUnitSearchOutcome(str, Enum):
    CHOOSE_CURRENT = "choose_current"
    SAMPLE_CURRENT_MORE = "sample_current_more"
    TRY_LARGER_UNIT = "try_larger_unit"


class GroundingObservableKind(str, Enum):
    PROGRAM_LOGIC = "program_logic"
    RUNTIME_VALUE_POINT = "runtime_value_point"
    BOUNDARY_CONTRACT = "boundary_contract"
    BOUNDARY_ARGUMENT_SIGNATURE = "boundary_argument_signature"


class RankSupportDomainKind(str, Enum):
    EXECUTION_DOMAIN = "execution_domain"
    REPLICATION_GROUP = "replication_group"


class RankSupportSplitKind(str, Enum):
    BRANCH_OUTCOME = "branch_outcome"
    LOCAL_POSITION_PARTITION = "local_position_partition"
    VALUE_PARTITION = "value_partition"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RankReductionEvidenceItem:
    kind: RankReductionEvidenceKind
    detail: str
    notes: tuple[str, ...] = ()


class RankSupportPartitionBasisKind(str, Enum):
    POSITIONAL = "positional"
    UNKNOWN = "unknown"


class RankSupportPartitionChildSelectorKind(str, Enum):
    POSITION_SET = "position_set"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RankSupportPartitionBasis:
    kind: RankSupportPartitionBasisKind
    source_ref: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankSupportPartitionChildSelector:
    kind: RankSupportPartitionChildSelectorKind
    label: str | None = None
    positions: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positions",
            tuple(sorted({int(position) for position in self.positions})),
        )


@dataclass(frozen=True)
class RankSupportPartitionChild:
    child_id: str
    count: int
    selector: RankSupportPartitionChildSelector | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = int(self.count)
        object.__setattr__(self, "count", count)
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")


@dataclass(frozen=True)
class RankSupportPartition:
    support_id: str
    parent_support_id: str | None = None
    domain_kind: RankSupportDomainKind = RankSupportDomainKind.EXECUTION_DOMAIN
    domain_size: int | None = None
    basis: RankSupportPartitionBasis = RankSupportPartitionBasis(
        kind=RankSupportPartitionBasisKind.UNKNOWN
    )
    children: tuple[RankSupportPartitionChild, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        domain_size = int(self.domain_size) if self.domain_size is not None else None
        object.__setattr__(self, "domain_size", domain_size)
        if domain_size is not None and domain_size <= 0:
            raise ValueError(f"domain_size must be positive when provided, got {domain_size}")

    @property
    def total_count(self) -> int:
        return sum(child.count for child in self.children)

    @property
    def nonempty_child_count(self) -> int:
        return sum(1 for child in self.children if child.count > 0)


@dataclass(frozen=True)
class GroundingObservable:
    observable_id: str
    kind: GroundingObservableKind
    source: str
    threshold_sensitive: bool = False
    minimum_grounding_unit_size: int = 1
    support_partition: RankSupportPartition | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        minimum_grounding_unit_size = int(self.minimum_grounding_unit_size)
        object.__setattr__(self, "minimum_grounding_unit_size", minimum_grounding_unit_size)
        if minimum_grounding_unit_size <= 0:
            raise ValueError(
                "minimum_grounding_unit_size must be positive, got "
                f"{minimum_grounding_unit_size}"
            )


@dataclass(frozen=True)
class GroundingObservableSamples:
    observable_id: str
    sample_values: tuple[str, ...]
    threshold_sensitive: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_values", tuple(str(item) for item in self.sample_values))

    @property
    def sample_count(self) -> int:
        return len(self.sample_values)


@dataclass(frozen=True)
class GroundingUnitSampleSet:
    grounding_unit_size: int
    replication_group_size: int
    observables: tuple[GroundingObservableSamples, ...]
    expressiveness_blocked: bool = False
    expressiveness_rationale: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        grounding_unit_size = int(self.grounding_unit_size)
        replication_group_size = int(self.replication_group_size)
        object.__setattr__(self, "grounding_unit_size", grounding_unit_size)
        object.__setattr__(self, "replication_group_size", replication_group_size)
        if grounding_unit_size <= 0:
            raise ValueError(f"grounding_unit_size must be positive, got {grounding_unit_size}")
        if replication_group_size <= 0:
            raise ValueError(f"replication_group_size must be positive, got {replication_group_size}")
        if grounding_unit_size > replication_group_size:
            raise ValueError(
                "grounding_unit_size cannot exceed replication_group_size: "
                f"{grounding_unit_size} > {replication_group_size}"
            )
        counts = {item.sample_count for item in self.observables}
        if len(counts) > 1:
            raise ValueError("all observables in one grounding-unit sample set must have the same sample count")

    @property
    def sample_count(self) -> int:
        if not self.observables:
            return 0
        return self.observables[0].sample_count


@dataclass(frozen=True)
class GroundingUnitSearchPolicy:
    min_samples_per_unit: int = 4
    stability_distance_threshold: float = 0.10
    threshold_sensitive_distance_threshold: float = 0.05
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingObservableStabilityAssessment:
    observable_id: str
    sample_count: int
    half_split_distance: float
    threshold_sensitive: bool = False
    stable: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingUnitSearchDecision:
    outcome: GroundingUnitSearchOutcome
    grounding_unit_size: int
    replication_group_size: int
    sample_count: int
    blocking_observable_ids: tuple[str, ...] = ()
    observable_assessments: tuple[GroundingObservableStabilityAssessment, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingUnitExpressivenessAssessment:
    grounding_unit_size: int
    replication_group_size: int
    expressiveness_blocked: bool
    blocking_observable_ids: tuple[str, ...] = ()
    observables: tuple[GroundingObservable, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingObservableSelection:
    changed_logic_point_names: tuple[str, ...]
    required_observables: tuple[GroundingObservable, ...]
    missing_observable_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingSelectionAssessment:
    selection: GroundingObservableSelection
    replication_group_size: int
    minimum_grounding_unit_lower_bound: int
    candidate_grounding_unit_sizes: tuple[int, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        replication_group_size = int(self.replication_group_size)
        lower_bound = int(self.minimum_grounding_unit_lower_bound)
        candidate_sizes = tuple(int(size) for size in self.candidate_grounding_unit_sizes)
        object.__setattr__(self, "replication_group_size", replication_group_size)
        object.__setattr__(self, "minimum_grounding_unit_lower_bound", lower_bound)
        object.__setattr__(self, "candidate_grounding_unit_sizes", candidate_sizes)
        if replication_group_size <= 0:
            raise ValueError(
                f"replication_group_size must be positive, got {replication_group_size}"
            )
        if lower_bound <= 0:
            raise ValueError(
                "minimum_grounding_unit_lower_bound must be positive, got "
                f"{lower_bound}"
            )
        if not candidate_sizes:
            raise ValueError("candidate_grounding_unit_sizes must not be empty")

    @property
    def changed_logic_point_names(self) -> tuple[str, ...]:
        return self.selection.changed_logic_point_names

    @property
    def required_observables(self) -> tuple[GroundingObservable, ...]:
        return self.selection.required_observables

    @property
    def missing_observable_ids(self) -> tuple[str, ...]:
        return self.selection.missing_observable_ids


@dataclass(frozen=True)
class AnchorRuntimeObservableEvidence:
    evidence_id: str
    observables: tuple[GroundingObservable, ...]
    source_logic_point_names: tuple[str, ...] = ()
    source_slice_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankReductionRequirementAssessment:
    required_requirements: tuple[RankReductionRequirementKind, ...]
    evidence_items: tuple[RankReductionEvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()

    def requires(self, requirement: RankReductionRequirementKind) -> bool:
        return RankReductionRequirementKind(requirement) in self.required_requirements


@dataclass(frozen=True)
class RankReductionPreservationContract:
    preserved_requirements: tuple[RankReductionRequirementKind, ...]
    unsupported_requirements: tuple[RankReductionRequirementKind, ...]
    evidence_items: tuple[RankReductionEvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()

    def preserves(self, requirement: RankReductionRequirementKind) -> bool:
        return RankReductionRequirementKind(requirement) in self.preserved_requirements


@dataclass(frozen=True)
class RankReductionEscalationDecision:
    outcome: RankReductionEscalationKind
    current_reduction_kind: RankReductionKind
    required_requirements: tuple[RankReductionRequirementKind, ...]
    missing_requirements: tuple[RankReductionRequirementKind, ...]
    recommended_reduction_kind: RankReductionKind | None = None
    recommended_rank_partitioned_behavior: bool | None = None
    recommended_grounding_unit_size: int | None = None
    recommended_replication_group_size: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def needs_full_spsd_dry_run(self) -> bool:
        return self.outcome == RankReductionEscalationKind.FULL_SPSD_DRY_RUN


@dataclass(frozen=True)
class RankReductionPlan:
    reduction_kind: RankReductionKind
    coverage_kind: RankReductionCoverageKind
    target_world_size: int
    execute_world_size: int
    representative_rank_groups: dict[int, tuple[int, ...]]
    representative_ranks: tuple[int, ...]
    portable_semantics: bool
    requires_target_cluster: bool
    replication_group_size: int | None = None
    preservation_contract: RankReductionPreservationContract | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        coverage_kind = RankReductionCoverageKind(self.coverage_kind)
        normalized_groups = {
            int(representative): tuple(int(rank) for rank in members)
            for representative, members in self.representative_rank_groups.items()
        }
        normalized_ranks = tuple(sorted(int(rank) for rank in self.representative_ranks))
        target_world_size = int(self.target_world_size)
        execute_world_size = int(self.execute_world_size)
        replication_group_size = (
            int(self.replication_group_size)
            if self.replication_group_size is not None
            else None
        )
        object.__setattr__(self, "coverage_kind", coverage_kind)
        object.__setattr__(self, "target_world_size", target_world_size)
        object.__setattr__(self, "execute_world_size", execute_world_size)
        object.__setattr__(self, "replication_group_size", replication_group_size)
        object.__setattr__(self, "representative_rank_groups", normalized_groups)
        object.__setattr__(self, "representative_ranks", normalized_ranks)
        if target_world_size <= 0:
            raise ValueError(f"target_world_size must be positive, got {target_world_size}")
        if execute_world_size <= 0:
            raise ValueError(f"execute_world_size must be positive, got {execute_world_size}")
        if execute_world_size > target_world_size:
            raise ValueError(
                "execute_world_size cannot exceed target_world_size: "
                f"{execute_world_size} > {target_world_size}"
            )
        if coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE:
            if replication_group_size is None or replication_group_size <= 0:
                raise ValueError(
                    "replication_group_size must be positive for replicated-group coverage"
                )
            if target_world_size % replication_group_size != 0:
                raise ValueError(
                    "target_world_size must be divisible by replication_group_size for replicated-group coverage: "
                    f"{target_world_size} % {replication_group_size}"
                )
            if execute_world_size > replication_group_size:
                raise ValueError(
                    "execute_world_size cannot exceed replication_group_size for replicated-group coverage: "
                    f"{execute_world_size} vs {replication_group_size}"
                )
        elif replication_group_size is not None and replication_group_size <= 0:
            raise ValueError(f"replication_group_size must be positive, got {replication_group_size}")
        expected_representatives = tuple(sorted(normalized_groups))
        if expected_representatives != normalized_ranks:
            raise ValueError(
                "representative_ranks must match representative_rank_groups keys: "
                f"{normalized_ranks} vs {expected_representatives}"
            )
        if coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE:
            for representative in normalized_groups:
                if representative < 0 or representative >= execute_world_size:
                    raise ValueError(
                        "replicated-group representatives must be executed local ranks within execute_world_size: "
                        f"{representative} not in [0, {execute_world_size})"
                    )
        covered: list[int] = []
        seen: set[int] = set()
        for representative, members in normalized_groups.items():
            if representative not in members:
                raise ValueError(
                    f"representative {representative} must appear in its own group {members}"
                )
            for rank in members:
                if rank in seen:
                    raise ValueError(f"rank {rank} appears in multiple representative groups")
                seen.add(rank)
                covered.append(rank)
        partition_world_size = (
            target_world_size
            if coverage_kind == RankReductionCoverageKind.TARGET_WORLD_COVERAGE
            else (
                replication_group_size
                if coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE
                else execute_world_size
            )
        )
        expected_covered = tuple(range(partition_world_size))
        actual_covered = tuple(sorted(covered))
        if actual_covered != expected_covered:
            raise ValueError(
                "representative_rank_groups must partition the declared coverage domain: "
                f"{actual_covered} vs {expected_covered}"
            )
        contract = self.preservation_contract
        if contract is None:
            contract = _default_preservation_contract(
                execute_world_size=execute_world_size,
                requires_target_cluster=bool(self.requires_target_cluster),
                coverage_kind=coverage_kind,
            )
        else:
            contract = RankReductionPreservationContract(
                preserved_requirements=tuple(
                    RankReductionRequirementKind(item)
                    for item in contract.preserved_requirements
                ),
                unsupported_requirements=tuple(
                    RankReductionRequirementKind(item)
                    for item in contract.unsupported_requirements
                ),
                evidence_items=tuple(
                    RankReductionEvidenceItem(
                        kind=RankReductionEvidenceKind(item.kind),
                        detail=str(item.detail),
                        notes=tuple(str(note) for note in item.notes),
                    )
                    for item in contract.evidence_items
                ),
                notes=tuple(str(note) for note in contract.notes),
            )
        object.__setattr__(self, "preservation_contract", contract)

    @property
    def profiled_world_size(self) -> int:
        return len(self.representative_ranks)

    @property
    def reduces_execution_world_size(self) -> bool:
        return self.execute_world_size < self.target_world_size

    @property
    def profiles_representative_subset(self) -> bool:
        return self.profiled_world_size < self.execute_world_size

    @property
    def coverage_world_size(self) -> int:
        if self.coverage_kind in {
            RankReductionCoverageKind.TARGET_WORLD_COVERAGE,
            RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE,
        }:
            return self.target_world_size
        return self.execute_world_size

    @property
    def covered_target_ranks(self) -> tuple[int, ...]:
        if self.coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE:
            return tuple(range(self.target_world_size))
        covered = sorted(
            {
                rank
                for members in self.representative_rank_groups.values()
                for rank in members
            }
        )
        return tuple(covered)

    @property
    def uncovered_target_ranks(self) -> tuple[int, ...]:
        covered = set(self.covered_target_ranks)
        return tuple(rank for rank in range(self.target_world_size) if rank not in covered)

    @property
    def target_coverage_fraction(self) -> float:
        return len(self.covered_target_ranks) / float(self.target_world_size)

    @property
    def has_full_target_coverage(self) -> bool:
        return len(self.uncovered_target_ranks) == 0

    def representative_for_target_rank(self, rank: int) -> int | None:
        target_rank = int(rank)
        if (
            self.coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE
            and self.replication_group_size is not None
        ):
            local_rank = target_rank % self.replication_group_size
            for representative, members in self.representative_rank_groups.items():
                if local_rank in members:
                    return representative
            return None
        for representative, members in self.representative_rank_groups.items():
            if target_rank in members:
                return representative
        return None

    def preserves(self, requirement: RankReductionRequirementKind) -> bool:
        return self.preservation_contract.preserves(requirement)


def _default_preservation_contract(
    *,
    execute_world_size: int,
    requires_target_cluster: bool,
    coverage_kind: RankReductionCoverageKind,
) -> RankReductionPreservationContract:
    preserved: list[RankReductionRequirementKind] = [
        RankReductionRequirementKind.PORTABLE_PROGRAM_LOGIC,
    ]
    notes: list[str] = [
        "Portable program logic is always preserved by rank reduction.",
    ]
    if execute_world_size > 1:
        preserved.append(RankReductionRequirementKind.RANK_PARTITION_BEHAVIOR)
        notes.append(
            "Multi-rank execution preserves upper-level rank-partition behavior on the covered domain."
        )
    if requires_target_cluster:
        preserved.append(RankReductionRequirementKind.TARGET_CLUSTER_REALIZATION)
        notes.append("Target-cluster realization is preserved for this reduction plan.")
    if coverage_kind == RankReductionCoverageKind.REPLICATED_GROUP_COVERAGE:
        preserved.append(RankReductionRequirementKind.REPLICATED_GROUP_COVERAGE)
        notes.append(
            "One executed hardware group is lifted across repeated homogeneous target groups."
        )
    if coverage_kind == RankReductionCoverageKind.TARGET_WORLD_COVERAGE:
        preserved.append(RankReductionRequirementKind.FULL_TARGET_WORLD_COVERAGE)
        notes.append("Representative groups cover the full target world.")
    unsupported = tuple(
        requirement
        for requirement in RankReductionRequirementKind
        if requirement not in preserved
    )
    return RankReductionPreservationContract(
        preserved_requirements=tuple(preserved),
        unsupported_requirements=unsupported,
        evidence_items=(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.POLICY_DEFAULT,
                detail="default reduction policy",
            ),
        ),
        notes=tuple(notes),
    )


def infer_rank_reduction_requirements(
    *,
    rank_partitioned_behavior: bool | None = None,
    open_obligations: tuple["Obligation", ...] = (),
    homogeneous_group_lift: bool = False,
    promoted_to_anchor: bool = False,
    affected_fraction: float | None = None,
    full_world_drift_threshold: float = 0.35,
) -> RankReductionRequirementAssessment:
    from .schema import ObligationEvidenceKind, ObligationKind, ObligationSourceKind

    requirements: list[RankReductionRequirementKind] = [
        RankReductionRequirementKind.PORTABLE_PROGRAM_LOGIC,
    ]
    evidence_items: list[RankReductionEvidenceItem] = [
        RankReductionEvidenceItem(
            kind=RankReductionEvidenceKind.POLICY_DEFAULT,
            detail="portable program logic is the base reduction requirement",
        )
    ]
    notes: list[str] = [
        "Portable program logic is always required before choosing a reduced execution.",
    ]
    if rank_partitioned_behavior is True:
        requirements.append(RankReductionRequirementKind.RANK_PARTITION_BEHAVIOR)
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.R_VARIANT_RANK_PARTITIONING,
                detail="old R-variant signal observed rank-partitioned opened-scope behavior",
                notes=(
                    "Used only as implementation evidence for requirement inference; not the paper abstraction.",
                ),
            )
        )
        notes.append(
            "Rank-partition evidence requires a semantic subset that preserves multi-rank behavior."
        )
    elif rank_partitioned_behavior is False:
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.R_VARIANT_WORLD_UNIFORM,
                detail="old R-variant signal stayed world-uniform",
                notes=(
                    "Used only as implementation evidence for requirement inference; not the paper abstraction.",
                ),
            )
        )
        notes.append(
            "World-uniform evidence leaves rank-partition behavior optional for this reduction choice."
        )
    cluster_obligations = tuple(
        obligation
        for obligation in open_obligations
        if obligation.kind in {ObligationKind.HARDWARE_BEHAVIOR, ObligationKind.CLUSTER_REALIZATION}
    )
    if cluster_obligations:
        requirements.append(RankReductionRequirementKind.TARGET_CLUSTER_REALIZATION)
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.OPEN_OBLIGATION_CLUSTER_REALIZATION,
                detail=";".join(
                    f"{obligation.kind.value}:{obligation.obligation_id}"
                    for obligation in cluster_obligations
                ),
                notes=(
                    "Open hardware or cluster-realization obligations require target-cluster grounding.",
                ),
            )
        )
        notes.append(
            "Open hardware/cluster obligations require target-cluster realization before the reduction is sufficient."
        )
    boundary_capsule_requirements: list[RankReductionRequirementKind] = []
    boundary_capsule_details: list[str] = []
    for obligation in open_obligations:
        if obligation.source_kind != ObligationSourceKind.BOUNDARY_CAPSULE:
            continue
        for item in obligation.evidence:
            if item.kind != ObligationEvidenceKind.PRESERVATION_REQUIREMENTS:
                continue
            requirement_names = tuple(
                name.strip() for name in item.value.split(",") if name.strip()
            )
            if not requirement_names:
                continue
            boundary_capsule_details.append(
                f"{obligation.obligation_id}:{','.join(requirement_names)}"
            )
            for requirement_name in requirement_names:
                boundary_capsule_requirements.append(
                    RankReductionRequirementKind(requirement_name)
                )
    if boundary_capsule_requirements:
        for requirement in boundary_capsule_requirements:
            requirements.append(requirement)
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.BOUNDARY_CAPSULE_REQUIREMENTS,
                detail=";".join(boundary_capsule_details),
                notes=(
                    "Boundary-contract incompatibilities can require stronger semantic preservation before reduced execution remains valid.",
                ),
            )
        )
        notes.append(
            "Boundary-contract preservation requirements were inferred from incompatible capsule obligations."
        )
    if homogeneous_group_lift:
        requirements.append(RankReductionRequirementKind.REPLICATED_GROUP_COVERAGE)
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.HOMOGENEOUS_GROUP_LIFT,
                detail="one real hardware group can be replicated across homogeneous target groups",
                notes=(
                    "Assumes one rank maps to one GPU and repeated target groups share the same physical structure.",
                ),
            )
        )
        notes.append(
            "Repeated homogeneous target groups permit one real hardware group to stand in for the full repeated group family."
        )
    if promoted_to_anchor and (affected_fraction or 0.0) >= full_world_drift_threshold:
        requirements.append(RankReductionRequirementKind.FULL_TARGET_WORLD_COVERAGE)
        evidence_items.append(
            RankReductionEvidenceItem(
                kind=RankReductionEvidenceKind.PROMOTED_HIGH_DRIFT,
                detail=(
                    f"promoted_high_drift:affected_fraction={float(affected_fraction or 0.0):.3f}"
                ),
                notes=(
                    "Large promoted drift requires representative full-world coverage before refresh is trusted.",
                ),
            )
        )
        notes.append(
            "High-drift promoted anchors require full target-world coverage before the reduced plan is sufficient."
        )
    normalized_requirements = tuple(dict.fromkeys(requirements))
    return RankReductionRequirementAssessment(
        required_requirements=normalized_requirements,
        evidence_items=tuple(evidence_items),
        notes=tuple(notes),
    )


def _infer_local_positions(
    ranks: tuple[int, ...],
    *,
    replication_group_size: int,
) -> tuple[int, ...]:
    if replication_group_size <= 0 or not ranks:
        return ()
    return tuple(sorted({int(rank) % int(replication_group_size) for rank in ranks}))


def _infer_positional_selectors(
    rank_groups: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    *,
    replication_group_size: int | None,
) -> tuple[RankSupportPartitionChildSelector, ...] | None:
    if replication_group_size is None:
        return None
    selectors: list[RankSupportPartitionChildSelector] = []
    seen_positions: set[int] = set()
    for group in rank_groups:
        local_positions = _infer_local_positions(
            tuple(int(rank) for rank in group),
            replication_group_size=int(replication_group_size),
        )
        if not local_positions:
            return None
        current_positions = set(local_positions)
        if current_positions & seen_positions:
            return None
        seen_positions.update(current_positions)
        selectors.append(
            RankSupportPartitionChildSelector(
                kind=RankSupportPartitionChildSelectorKind.POSITION_SET,
                positions=local_positions,
                notes=(
                    "selector derived from stable local positions within the replication group",
                ),
            )
        )
    if not selectors:
        return None
    return tuple(selectors)


def rank_support_partition_from_rank_groups(
    *,
    support_id: str,
    rank_groups: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    parent_support_id: str | None = None,
    domain_kind: RankSupportDomainKind = RankSupportDomainKind.EXECUTION_DOMAIN,
    domain_size: int | None = None,
    split_kind: RankSupportSplitKind = RankSupportSplitKind.BRANCH_OUTCOME,
    replication_group_size: int | None = None,
    child_prefix: str = "child",
    source_ref: str | None = None,
    notes: tuple[str, ...] = (),
) -> RankSupportPartition | None:
    del split_kind  # structural support partitions are positional-only
    selectors = _infer_positional_selectors(
        rank_groups,
        replication_group_size=replication_group_size,
    )
    if selectors is None:
        return None
    children: list[RankSupportPartitionChild] = []
    for index, group in enumerate(rank_groups):
        normalized_group = tuple(int(rank) for rank in group)
        children.append(
            RankSupportPartitionChild(
                child_id=f"{child_prefix}_{index}",
                count=len(normalized_group),
                selector=selectors[index],
            )
        )
    return RankSupportPartition(
        support_id=support_id,
        parent_support_id=parent_support_id,
        domain_kind=domain_kind,
        domain_size=domain_size,
        basis=RankSupportPartitionBasis(
            kind=RankSupportPartitionBasisKind.POSITIONAL,
            source_ref=source_ref,
            notes=(
                "structural support partitions are restricted to stable local-position partitions",
            ),
        ),
        children=tuple(children),
        notes=tuple(notes),
    )


def rank_support_partition_from_logic_point(
    point: "ProgramLogicPoint",
    *,
    parent_support_id: str | None = None,
    domain_size: int | None = None,
    replication_group_size: int | None = None,
) -> RankSupportPartition | None:
    if not point.rank_groups:
        return None
    if replication_group_size is None:
        return None
    return rank_support_partition_from_rank_groups(
        support_id=f"support:{point.name}",
        rank_groups=point.rank_groups,
        parent_support_id=parent_support_id,
        domain_kind=RankSupportDomainKind.REPLICATION_GROUP,
        domain_size=domain_size,
        split_kind=RankSupportSplitKind.LOCAL_POSITION_PARTITION,
        replication_group_size=replication_group_size,
        child_prefix=point.name,
        source_ref=f"logic:{point.name}",
        notes=(
            "support partition derived from program-logic rank groups",
            "exact rank ids are intentionally abstracted away in favor of child counts and stable local-position selectors",
            "non-positional runtime differences stay in runtime-value observables instead of the structural partition object",
        ),
    )


def grounding_observables_from_program_logic(
    points: tuple["ProgramLogicPoint", ...] | list["ProgramLogicPoint"],
    *,
    support_domain_size: int | None = None,
    replication_group_size: int | None = None,
) -> tuple[GroundingObservable, ...]:
    from .program_logic import ProgramLogicKind

    observables: list[GroundingObservable] = []
    for point in points:
        minimum_grounding_unit_size = 1
        notes: list[str] = list(point.notes)
        support_partition = rank_support_partition_from_logic_point(
            point,
            domain_size=support_domain_size,
            replication_group_size=replication_group_size,
        )
        if point.kind == ProgramLogicKind.RANK_PARTITION:
            observed_support_classes = (
                support_partition.nonempty_child_count
                if support_partition is not None
                else sum(1 for group in point.rank_groups if group)
            )
            minimum_grounding_unit_size = max(
                1,
                observed_support_classes or 2,
            )
            if minimum_grounding_unit_size <= 1:
                notes.append(
                    "Observed rank groups remain world-uniform, so one local position is enough unless runtime-value observables demand more."
                )
            else:
                notes.append(
                    "Observed rank groups create multiple active support classes, so the grounding unit must cover at least that many local positions before it can represent divergent rank behavior."
                )
        elif point.rank_groups and len(point.rank_groups) > 1:
            minimum_grounding_unit_size = 2
            notes.append(
                "Multiple observed rank groups indicate that one local position is unlikely to cover all value-sensitive outcomes."
            )
        observables.append(
            GroundingObservable(
                observable_id=f"logic:{point.name}",
                kind=GroundingObservableKind.PROGRAM_LOGIC,
                source=point.source,
                threshold_sensitive=True,
                minimum_grounding_unit_size=minimum_grounding_unit_size,
                support_partition=support_partition,
                notes=tuple(notes),
            )
        )
    return tuple(observables)


def grounding_observables_from_runtime_value_points(
    points: tuple["RuntimeValuePoint", ...] | list["RuntimeValuePoint"],
) -> tuple[GroundingObservable, ...]:
    from .semantic_basis import RuntimeValuePointKind

    observables: list[GroundingObservable] = []
    for point in points:
        required_field_names = tuple(point.required_fields) + tuple(point.inferable_fields)
        minimum_grounding_unit_size = 1
        threshold_sensitive = point.kind in {
            RuntimeValuePointKind.BRANCH,
            RuntimeValuePointKind.COUNT,
            RuntimeValuePointKind.BUCKET,
        }
        notes: list[str] = list(point.notes)
        if point.kind in {RuntimeValuePointKind.COUNT, RuntimeValuePointKind.BUCKET} and any(
            field.startswith("parallelism.") or "dispatch_footprint" in field
            for field in required_field_names
        ):
            minimum_grounding_unit_size = 2
            notes.append(
                "This runtime-value point depends on cross-rank placement or dispatch summaries, so a single local position may be too weak."
            )
        observables.append(
            GroundingObservable(
                observable_id=f"runtime:{point.name}",
                kind=GroundingObservableKind.RUNTIME_VALUE_POINT,
                source=point.name,
                threshold_sensitive=threshold_sensitive,
                minimum_grounding_unit_size=minimum_grounding_unit_size,
                notes=tuple(notes),
            )
        )
    return tuple(observables)


def grounding_observables_from_boundary_contracts(
    contracts: tuple["BoundaryContract", ...] | list["BoundaryContract"],
) -> tuple[GroundingObservable, ...]:
    observables: list[GroundingObservable] = []
    for index, contract in enumerate(contracts):
        source = f"{contract.callee_name}:{contract.boundary_kind}"
        summary_notes = tuple(contract.notes) + (
            "Boundary-contract observables track execution-visible call semantics, not opaque library internals.",
        )
        observables.append(
            GroundingObservable(
                observable_id=f"boundary:{index}:{contract.callee_name}",
                kind=GroundingObservableKind.BOUNDARY_CONTRACT,
                source=source,
                threshold_sensitive=bool(
                    contract.logic_observable
                    or contract.mutates_state
                    or contract.advances_group_state
                ),
                minimum_grounding_unit_size=1,
                notes=summary_notes,
            )
        )
        if contract.positional_arg_kinds or contract.keyword_arg_names:
            observables.append(
                GroundingObservable(
                    observable_id=f"boundary_args:{index}:{contract.callee_name}",
                    kind=GroundingObservableKind.BOUNDARY_ARGUMENT_SIGNATURE,
                    source=source,
                    threshold_sensitive=bool(contract.logic_observable),
                    minimum_grounding_unit_size=1,
                    notes=(
                        "Argument-signature observables summarize execution-relevant boundary-call shapes and flags.",
                    ),
                )
            )
    return tuple(observables)


def derive_grounding_observables(
    *,
    program_logic_points: tuple["ProgramLogicPoint", ...] | list["ProgramLogicPoint"] = (),
    runtime_value_points: tuple["RuntimeValuePoint", ...] | list["RuntimeValuePoint"] = (),
    boundary_contracts: tuple["BoundaryContract", ...] | list["BoundaryContract"] = (),
    replication_group_size: int | None = None,
) -> tuple[GroundingObservable, ...]:
    observables = (
        *grounding_observables_from_program_logic(
            program_logic_points,
            replication_group_size=replication_group_size,
        ),
        *grounding_observables_from_runtime_value_points(runtime_value_points),
        *grounding_observables_from_boundary_contracts(boundary_contracts),
    )
    deduped: list[GroundingObservable] = []
    seen: set[str] = set()
    for observable in observables:
        if observable.observable_id in seen:
            continue
        seen.add(observable.observable_id)
        deduped.append(observable)
    return tuple(deduped)


def assess_grounding_unit_expressiveness(
    observables: tuple[GroundingObservable, ...] | list[GroundingObservable],
    *,
    grounding_unit_size: int,
    replication_group_size: int,
) -> GroundingUnitExpressivenessAssessment:
    actual_grounding_unit_size = int(grounding_unit_size)
    actual_replication_group_size = int(replication_group_size)
    if actual_grounding_unit_size <= 0:
        raise ValueError(
            f"grounding_unit_size must be positive, got {actual_grounding_unit_size}"
        )
    if actual_replication_group_size <= 0:
        raise ValueError(
            "replication_group_size must be positive, got "
            f"{actual_replication_group_size}"
        )
    if actual_grounding_unit_size > actual_replication_group_size:
        raise ValueError(
            "grounding_unit_size cannot exceed replication_group_size: "
            f"{actual_grounding_unit_size} > {actual_replication_group_size}"
        )
    blocking = tuple(
        observable.observable_id
        for observable in observables
        if observable.minimum_grounding_unit_size > actual_grounding_unit_size
    )
    notes: list[str] = []
    if blocking:
        notes.append(
            "Some execution-relevant observables require more local positions than the current grounding unit can represent."
        )
    else:
        notes.append(
            "The current grounding unit can represent the minimum local-position diversity required by the tracked observables."
        )
    return GroundingUnitExpressivenessAssessment(
        grounding_unit_size=actual_grounding_unit_size,
        replication_group_size=actual_replication_group_size,
        expressiveness_blocked=bool(blocking),
        blocking_observable_ids=blocking,
        observables=tuple(observables),
        notes=tuple(notes),
    )


def minimum_grounding_unit_lower_bound(
    observables: tuple[GroundingObservable, ...] | list[GroundingObservable],
    *,
    replication_group_size: int | None = None,
) -> int:
    lower_bound = max(
        (int(observable.minimum_grounding_unit_size) for observable in observables),
        default=1,
    )
    if replication_group_size is None:
        return lower_bound
    actual_replication_group_size = int(replication_group_size)
    if actual_replication_group_size <= 0:
        raise ValueError(
            f"replication_group_size must be positive, got {actual_replication_group_size}"
        )
    if lower_bound > actual_replication_group_size:
        raise ValueError(
            "minimum grounding-unit lower bound cannot exceed replication_group_size: "
            f"{lower_bound} > {actual_replication_group_size}"
        )
    return lower_bound


def recommended_grounding_unit_sizes(
    observables: tuple[GroundingObservable, ...] | list[GroundingObservable],
    *,
    replication_group_size: int,
) -> tuple[int, ...]:
    actual_replication_group_size = int(replication_group_size)
    if actual_replication_group_size <= 0:
        raise ValueError(
            f"replication_group_size must be positive, got {actual_replication_group_size}"
        )
    lower_bound = minimum_grounding_unit_lower_bound(
        observables,
        replication_group_size=actual_replication_group_size,
    )
    sizes: list[int] = [lower_bound]
    current = lower_bound
    while current < actual_replication_group_size:
        next_size = min(actual_replication_group_size, max(current + 1, current * 2))
        if next_size == current:
            break
        sizes.append(next_size)
        current = next_size
    return tuple(sizes)


def assess_grounding_selection(
    selection: GroundingObservableSelection,
    *,
    replication_group_size: int,
) -> GroundingSelectionAssessment:
    actual_replication_group_size = int(replication_group_size)
    lower_bound = minimum_grounding_unit_lower_bound(
        selection.required_observables,
        replication_group_size=actual_replication_group_size,
    )
    candidate_sizes = recommended_grounding_unit_sizes(
        selection.required_observables,
        replication_group_size=actual_replication_group_size,
    )
    notes = [
        "Grounding selection packages mandatory observables, explicit observable gaps, and the induced grounding-size ladder together for planning."
    ]
    if selection.missing_observable_ids:
        notes.append(
            "The grounding selection still has explicit observable gaps: "
            + ",".join(selection.missing_observable_ids)
        )
    return GroundingSelectionAssessment(
        selection=selection,
        replication_group_size=actual_replication_group_size,
        minimum_grounding_unit_lower_bound=lower_bound,
        candidate_grounding_unit_sizes=candidate_sizes,
        notes=tuple(dict.fromkeys((*selection.notes, *notes))),
    )


def _required_runtime_value_point_names_for_logic_kind(
    logic_kind: "ProgramLogicKind",
) -> tuple[str, ...]:
    from .program_logic import ProgramLogicKind

    mapping: dict[ProgramLogicKind, tuple[str, ...]] = {
        ProgramLogicKind.RANK_PARTITION: (
            "route_decision",
            "expert_load",
            "remote_dispatch",
        ),
        ProgramLogicKind.REGION_ACTIVATION: (
            "overflow_state",
            "expert_load",
        ),
        ProgramLogicKind.STAGE_ORDER: ("collective_stage",),
    }
    return mapping.get(ProgramLogicKind(logic_kind), ())


def _runtime_value_point_names_from_selected_fields(
    selected_points: tuple["ProgramLogicPoint", ...] | list["ProgramLogicPoint"],
    runtime_value_points: tuple["RuntimeValuePoint", ...] | list["RuntimeValuePoint"],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_fields = {
        str(field)
        for point in selected_points
        for field in getattr(point, "source_fields", ())
    }
    if not selected_fields:
        return (), ()
    derived: list[str] = []
    for point in runtime_value_points:
        point_fields = set(point.required_fields) | set(point.inferable_fields)
        if point_fields.intersection(selected_fields) and point.name not in derived:
            derived.append(point.name)
    return tuple(derived), tuple(sorted(selected_fields))


def _boundary_contract_relevant_to_logic_kind(
    contract: "BoundaryContract",
    logic_kind: "ProgramLogicKind",
) -> bool:
    from .program_logic import ProgramLogicKind

    normalized_kind = ProgramLogicKind(logic_kind)
    if normalized_kind == ProgramLogicKind.RANK_PARTITION:
        return bool(
            contract.boundary_kind in {"collective", "dispatch"}
            or contract.logic_observable
            or contract.advances_group_state
        )
    if normalized_kind == ProgramLogicKind.REGION_ACTIVATION:
        return bool(contract.logic_observable or contract.mutates_state)
    if normalized_kind == ProgramLogicKind.STAGE_ORDER:
        return bool(
            contract.boundary_kind == "collective" or contract.advances_group_state
        )
    return False


def _partition_anchor_runtime_observable_evidence(
    evidence_items: tuple[AnchorRuntimeObservableEvidence, ...] | list[AnchorRuntimeObservableEvidence],
    *,
    changed_logic_point_names: tuple[str, ...],
    changed_slice_ids: tuple[str, ...],
) -> tuple[tuple[AnchorRuntimeObservableEvidence, ...], tuple[AnchorRuntimeObservableEvidence, ...]]:
    changed_name_set = set(changed_logic_point_names)
    changed_slice_set = set(changed_slice_ids)
    reusable: list[AnchorRuntimeObservableEvidence] = []
    dropped: list[AnchorRuntimeObservableEvidence] = []
    for item in evidence_items:
        if set(item.source_logic_point_names).intersection(changed_name_set) or set(
            item.source_slice_ids
        ).intersection(changed_slice_set):
            dropped.append(item)
        else:
            reusable.append(item)
    return tuple(reusable), tuple(dropped)


def select_mandatory_grounding_observables(
    *,
    changed_program_logic_points: tuple["ProgramLogicPoint", ...] | list["ProgramLogicPoint"],
    all_program_logic_points: tuple["ProgramLogicPoint", ...] | list["ProgramLogicPoint"] = (),
    runtime_value_points: tuple["RuntimeValuePoint", ...] | list["RuntimeValuePoint"] = (),
    boundary_contracts: tuple["BoundaryContract", ...] | list["BoundaryContract"] = (),
    boundary_capsules: tuple["BoundaryContextCapsule", ...] | list["BoundaryContextCapsule"] = (),
    emission_dag: "EmissionDAG | None" = None,
    logic_point_ownership_map: "LogicPointOwnershipMap | None" = None,
    emission_ownership_map: "EmissionOwnershipMap | None" = None,
    logic_state_store: "LogicStateStore | None" = None,
    logic_slice_graph: "LogicSliceGraph | None" = None,
    anchor_runtime_observable_evidence: tuple[AnchorRuntimeObservableEvidence, ...]
    | list[AnchorRuntimeObservableEvidence] = (),
) -> GroundingObservableSelection:
    changed_points = tuple(changed_program_logic_points)
    changed_names = tuple(point.name for point in changed_points)
    all_points = tuple(all_program_logic_points) or changed_points
    point_by_name = {point.name: point for point in all_points}
    selected_points_by_name: dict[str, ProgramLogicPoint] = {
        point.name: point for point in changed_points
    }
    selected_region_ids: set[str] = set()
    selected_cfg_block_ids: set[str] = set()
    selected_slice_ids: set[str] = set()
    if logic_point_ownership_map is not None:
        for point in changed_points:
            ownership = logic_point_ownership_map.ownership_for_point(point.name)
            if ownership is None:
                continue
            selected_region_ids.update(ownership.owning_region_ids)
            selected_cfg_block_ids.update(ownership.owning_cfg_block_ids)
    if logic_state_store is not None:
        successor_map = logic_slice_graph.successor_map() if logic_slice_graph is not None else {}
        slice_by_id = logic_state_store.slice_by_id()
        for slice_state in logic_state_store.slices:
            if (
                set(slice_state.logic_point_names).intersection(changed_names)
                or set(slice_state.owned_region_ids).intersection(selected_region_ids)
                or set(slice_state.owned_cfg_block_ids).intersection(selected_cfg_block_ids)
            ):
                selected_slice_ids.add(slice_state.slice_id)
        queue = list(selected_slice_ids)
        while queue:
            src_slice_id = queue.pop(0)
            for dst_slice_id in successor_map.get(src_slice_id, ()):
                if dst_slice_id in selected_slice_ids:
                    continue
                successor = slice_by_id.get(dst_slice_id)
                if successor is None or successor.checkpointable:
                    continue
                selected_slice_ids.add(dst_slice_id)
                queue.append(dst_slice_id)
        for slice_id in tuple(selected_slice_ids):
            slice_state = slice_by_id.get(slice_id)
            if slice_state is None:
                continue
            selected_region_ids.update(slice_state.owned_region_ids)
            selected_cfg_block_ids.update(slice_state.owned_cfg_block_ids)
            for point_name in slice_state.logic_point_names:
                point = point_by_name.get(point_name)
                if point is not None:
                    selected_points_by_name.setdefault(point_name, point)
    if logic_point_ownership_map is not None and (selected_region_ids or selected_cfg_block_ids):
        for point in all_points:
            if point.name in selected_points_by_name:
                continue
            ownership = logic_point_ownership_map.ownership_for_point(point.name)
            if ownership is None:
                continue
            if set(ownership.owning_region_ids).intersection(selected_region_ids) or set(
                ownership.owning_cfg_block_ids
            ).intersection(selected_cfg_block_ids):
                selected_points_by_name[point.name] = point
    selected_points = tuple(selected_points_by_name.values())
    selected: list[GroundingObservable] = list(
        grounding_observables_from_program_logic(selected_points)
    )
    runtime_value_by_name = {
        point.name: point for point in runtime_value_points
    }
    fallback_boundary_contracts: list[BoundaryContract] = list(boundary_contracts)
    structurally_selected_boundary_contracts: list[BoundaryContract] = []
    if boundary_capsules:
        stub_to_emission_id = (
            {node.stub_id: node.emission_id for node in emission_dag.nodes}
            if emission_dag is not None
            else {}
        )
        for capsule in boundary_capsules:
            stub_emission_id = stub_to_emission_id.get(capsule.stub_id)
            matched = False
            if stub_emission_id is not None and logic_state_store is not None:
                for slice_state in logic_state_store.slices:
                    if stub_emission_id in slice_state.emission_ids and slice_state.slice_id in selected_slice_ids:
                        matched = True
                        break
            if (
                not matched
                and emission_ownership_map is not None
                and stub_emission_id is not None
            ):
                ownership = emission_ownership_map.ownership_for_emission(stub_emission_id)
                if ownership is not None and (
                    set(ownership.owning_region_ids).intersection(selected_region_ids)
                    or set(ownership.owning_cfg_block_ids).intersection(selected_cfg_block_ids)
                ):
                    matched = True
            if matched:
                structurally_selected_boundary_contracts.append(capsule.boundary_contract)
    reusable_runtime_evidence, dropped_runtime_evidence = _partition_anchor_runtime_observable_evidence(
        anchor_runtime_observable_evidence,
        changed_logic_point_names=changed_names,
        changed_slice_ids=tuple(sorted(selected_slice_ids)),
    )
    runtime_observable_from_evidence = {
        observable.observable_id: observable
        for evidence in reusable_runtime_evidence
        for observable in evidence.observables
        if observable.kind == GroundingObservableKind.RUNTIME_VALUE_POINT
    }
    fallback_boundary_observables = grounding_observables_from_boundary_contracts(
        fallback_boundary_contracts
    )
    structural_boundary_observables = grounding_observables_from_boundary_contracts(
        structurally_selected_boundary_contracts
    )
    selected_ids = {observable.observable_id for observable in selected}
    missing_ids: list[str] = []
    notes: list[str] = [
        "Mandatory observables are selected conservatively from the changed logic region and the execution-visible values/boundaries it can influence."
    ]
    if selected_slice_ids:
        notes.append(
            "Changed-region selection is anchored to the actual owned logic slices and expands only through non-checkpointable successor slices."
        )
    field_derived_runtime_point_names, selected_source_fields = _runtime_value_point_names_from_selected_fields(
        selected_points,
        runtime_value_points,
    )
    fallback_logic_kinds: list[str] = []
    for point in selected_points:
        runtime_point_names = field_derived_runtime_point_names
        if not runtime_point_names:
            runtime_point_names = _required_runtime_value_point_names_for_logic_kind(point.kind)
            fallback_logic_kinds.append(point.kind.value)
        for runtime_point_name in runtime_point_names:
            runtime_point = runtime_value_by_name.get(runtime_point_name)
            observable_id = f"runtime:{runtime_point_name}"
            if runtime_point is None:
                reusable_observable = runtime_observable_from_evidence.get(observable_id)
                if reusable_observable is not None:
                    if reusable_observable.observable_id not in selected_ids:
                        selected.append(reusable_observable)
                        selected_ids.add(reusable_observable.observable_id)
                    notes.append(
                        f"{point.name} reuses anchor-derived runtime observable {observable_id} because its producing region is unchanged."
                    )
                    continue
                if observable_id not in missing_ids:
                    missing_ids.append(observable_id)
                continue
            for observable in grounding_observables_from_runtime_value_points((runtime_point,)):
                if observable.observable_id not in selected_ids:
                    selected.append(observable)
                    selected_ids.add(observable.observable_id)
        for contract_index, contract in enumerate(fallback_boundary_contracts):
            if not _boundary_contract_relevant_to_logic_kind(contract, point.kind):
                continue
            prefix = f"{contract.callee_name}:{contract.boundary_kind}"
            for observable in fallback_boundary_observables:
                if observable.source != prefix:
                    continue
                if observable.observable_id not in selected_ids:
                    selected.append(observable)
                    selected_ids.add(observable.observable_id)
            notes.append(
                f"{point.name} pulls in boundary contract {contract_index}:{prefix} through conservative logic-to-boundary dependence."
            )
    if structurally_selected_boundary_contracts:
        for contract_index, contract in enumerate(structurally_selected_boundary_contracts):
            prefix = f"{contract.callee_name}:{contract.boundary_kind}"
            for observable in structural_boundary_observables:
                if observable.source != prefix:
                    continue
                if observable.observable_id not in selected_ids:
                    selected.append(observable)
                    selected_ids.add(observable.observable_id)
            notes.append(
                f"Changed-region ownership pulls in boundary contract {contract_index}:{prefix} directly from selected slices/emissions."
            )
    if selected_source_fields:
        notes.append(
            "Runtime-value observables are derived from selected logic-point source fields: "
            + ",".join(selected_source_fields)
        )
    if fallback_logic_kinds:
        notes.append(
            "Some selected logic points do not expose source fields, so the selector fell back to conservative logic-kind runtime-point mapping for: "
            + ",".join(sorted(dict.fromkeys(fallback_logic_kinds)))
        )
    if missing_ids:
        notes.append(
            "Some required runtime-value observables are not yet available and remain explicit gaps in the dependence slice."
        )
    if dropped_runtime_evidence:
        notes.append(
            "Anchor-derived runtime observables from changed logic points or changed structural slices are not reused."
        )
    return GroundingObservableSelection(
        changed_logic_point_names=changed_names,
        required_observables=tuple(selected),
        missing_observable_ids=tuple(missing_ids),
        notes=tuple(dict.fromkeys(notes)),
    )


def _distribution_from_samples(samples: tuple[str, ...]) -> dict[str, float]:
    if not samples:
        return {}
    counts: dict[str, int] = {}
    for item in samples:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(samples))
    return {key: value / total for key, value in counts.items()}


def _total_variation_distance(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def assess_grounding_unit_sampling(
    sample_set: GroundingUnitSampleSet,
    *,
    policy: GroundingUnitSearchPolicy | None = None,
) -> GroundingUnitSearchDecision:
    resolved_policy = policy or GroundingUnitSearchPolicy()
    sample_count = sample_set.sample_count
    if sample_count < resolved_policy.min_samples_per_unit:
        return GroundingUnitSearchDecision(
            outcome=GroundingUnitSearchOutcome.SAMPLE_CURRENT_MORE,
            grounding_unit_size=sample_set.grounding_unit_size,
            replication_group_size=sample_set.replication_group_size,
            sample_count=sample_count,
            notes=(
                "repeat sampling on the current grounding unit before trying a larger unit",
            ),
        )
    observable_assessments: list[GroundingObservableStabilityAssessment] = []
    blocking_observable_ids: list[str] = []
    for observable in sample_set.observables:
        midpoint = observable.sample_count // 2
        left = observable.sample_values[:midpoint]
        right = observable.sample_values[midpoint:]
        distance = _total_variation_distance(
            _distribution_from_samples(left),
            _distribution_from_samples(right),
        )
        threshold = (
            resolved_policy.threshold_sensitive_distance_threshold
            if observable.threshold_sensitive
            else resolved_policy.stability_distance_threshold
        )
        stable = distance <= threshold
        assessment = GroundingObservableStabilityAssessment(
            observable_id=observable.observable_id,
            sample_count=observable.sample_count,
            half_split_distance=distance,
            threshold_sensitive=observable.threshold_sensitive,
            stable=stable,
            notes=observable.notes,
        )
        observable_assessments.append(assessment)
        if not stable:
            blocking_observable_ids.append(observable.observable_id)
    if sample_set.expressiveness_blocked:
        return GroundingUnitSearchDecision(
            outcome=GroundingUnitSearchOutcome.TRY_LARGER_UNIT,
            grounding_unit_size=sample_set.grounding_unit_size,
            replication_group_size=sample_set.replication_group_size,
            sample_count=sample_count,
            blocking_observable_ids=tuple(blocking_observable_ids),
            observable_assessments=tuple(observable_assessments),
            notes=(
                *sample_set.expressiveness_rationale,
                "the current grounding unit is structurally unable to represent all execution-relevant variability",
            ),
        )
    if blocking_observable_ids:
        return GroundingUnitSearchDecision(
            outcome=GroundingUnitSearchOutcome.SAMPLE_CURRENT_MORE,
            grounding_unit_size=sample_set.grounding_unit_size,
            replication_group_size=sample_set.replication_group_size,
            sample_count=sample_count,
            blocking_observable_ids=tuple(blocking_observable_ids),
            observable_assessments=tuple(observable_assessments),
            notes=(
                "execution-relevant observables have not stabilized yet; repeat sampling on the current grounding unit",
            ),
        )
    return GroundingUnitSearchDecision(
        outcome=GroundingUnitSearchOutcome.CHOOSE_CURRENT,
        grounding_unit_size=sample_set.grounding_unit_size,
        replication_group_size=sample_set.replication_group_size,
        sample_count=sample_count,
        observable_assessments=tuple(observable_assessments),
        notes=(
            "execution-relevant observables are stable on the current grounding unit",
        ),
    )


def choose_minimum_grounding_unit(
    sample_sets: tuple[GroundingUnitSampleSet, ...] | list[GroundingUnitSampleSet],
    *,
    policy: GroundingUnitSearchPolicy | None = None,
) -> GroundingUnitSearchDecision:
    resolved_policy = policy or GroundingUnitSearchPolicy()
    ordered = sorted(
        sample_sets,
        key=lambda item: (item.grounding_unit_size, item.replication_group_size),
    )
    last_decision: GroundingUnitSearchDecision | None = None
    for sample_set in ordered:
        decision = assess_grounding_unit_sampling(sample_set, policy=resolved_policy)
        last_decision = decision
        if decision.outcome != GroundingUnitSearchOutcome.TRY_LARGER_UNIT:
            return decision
    if last_decision is not None:
        return last_decision
    raise ValueError("at least one grounding-unit sample set is required")
def _minimal_reduction_for_requirements(
    requirements: tuple[RankReductionRequirementKind, ...],
    *,
    grounding_unit_size: int | None = None,
    replication_group_size: int | None = None,
) -> tuple[RankReductionKind | None, bool | None, int | None, int | None]:
    required_set = {RankReductionRequirementKind(item) for item in requirements}
    if RankReductionRequirementKind.EXACT_PER_RANK_REALIZATION in required_set:
        return None, None, None, None
    if RankReductionRequirementKind.FULL_TARGET_WORLD_COVERAGE in required_set:
        return RankReductionKind.REPRESENTATIVE_FULL_WORLD, None, None, None
    if RankReductionRequirementKind.REPLICATED_GROUP_COVERAGE in required_set:
        recommended_replication_group_size = int(replication_group_size or 8)
        recommended_grounding_unit_size = int(
            grounding_unit_size or recommended_replication_group_size
        )
        return (
            RankReductionKind.GROUP_REPLICATED_SUBSET,
            None,
            recommended_grounding_unit_size,
            recommended_replication_group_size,
        )
    if RankReductionRequirementKind.TARGET_CLUSTER_REALIZATION in required_set:
        return RankReductionKind.CLUSTER_CALIBRATION_SUBSET, None, None, None
    if RankReductionRequirementKind.RANK_PARTITION_BEHAVIOR in required_set:
        return RankReductionKind.SEMANTIC_SUBSET, True, None, None
    if RankReductionRequirementKind.PORTABLE_PROGRAM_LOGIC in required_set:
        return RankReductionKind.SEMANTIC_SUBSET, None, None, None
    return RankReductionKind.SEMANTIC_SUBSET, None, None, None


def assess_rank_reduction_escalation(
    plan: RankReductionPlan,
    *,
    required_requirements: tuple[RankReductionRequirementKind, ...] | list[RankReductionRequirementKind],
    grounding_unit_size: int | None = None,
    replication_group_size: int | None = None,
) -> RankReductionEscalationDecision:
    normalized_requirements = tuple(
        dict.fromkeys(RankReductionRequirementKind(item) for item in required_requirements)
    )
    missing = tuple(
        requirement
        for requirement in normalized_requirements
        if not plan.preserves(requirement)
    )
    if not missing:
        return RankReductionEscalationDecision(
            outcome=RankReductionEscalationKind.SATISFIED,
            current_reduction_kind=plan.reduction_kind,
            required_requirements=normalized_requirements,
            missing_requirements=(),
            recommended_reduction_kind=plan.reduction_kind,
            notes=("Current reduction already preserves the required semantics.",),
        )
    (
        recommended_kind,
        recommended_rank_partitioned_behavior,
        recommended_grounding_unit_size,
        recommended_replication_group_size,
    ) = _minimal_reduction_for_requirements(
        missing,
        grounding_unit_size=grounding_unit_size,
        replication_group_size=replication_group_size,
    )
    if recommended_kind is None:
        return RankReductionEscalationDecision(
            outcome=RankReductionEscalationKind.FULL_SPSD_DRY_RUN,
            current_reduction_kind=plan.reduction_kind,
            required_requirements=normalized_requirements,
            missing_requirements=missing,
            notes=(
                "Required semantics exceed the strongest reduced mode and need exact per-rank realization.",
            ),
        )
    notes: list[str] = []
    if (
        recommended_kind == RankReductionKind.SEMANTIC_SUBSET
        and plan.reduction_kind == RankReductionKind.SEMANTIC_SUBSET
        and recommended_rank_partitioned_behavior is True
        and not plan.preserves(RankReductionRequirementKind.RANK_PARTITION_BEHAVIOR)
    ):
        notes.append(
            "Widen the semantic subset from world-uniform single-rank grounding to multi-rank partition-preserving grounding."
        )
    else:
        notes.append(
            f"Escalate from {plan.reduction_kind.value} to {recommended_kind.value} to preserve the missing requirements."
        )
    if (
        recommended_kind == RankReductionKind.GROUP_REPLICATED_SUBSET
        and recommended_replication_group_size is not None
    ):
        notes.append(
            "Lift one real grounding unit "
            f"of size {recommended_grounding_unit_size or recommended_replication_group_size} "
            f"across repeated homogeneous target groups of size {recommended_replication_group_size}."
        )
    return RankReductionEscalationDecision(
        outcome=RankReductionEscalationKind.STRENGTHEN_REDUCTION,
        current_reduction_kind=plan.reduction_kind,
        required_requirements=normalized_requirements,
        missing_requirements=missing,
        recommended_reduction_kind=recommended_kind,
        recommended_rank_partitioned_behavior=recommended_rank_partitioned_behavior,
        recommended_grounding_unit_size=recommended_grounding_unit_size,
        recommended_replication_group_size=recommended_replication_group_size,
        notes=tuple(notes),
    )
