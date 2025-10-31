"""Deterministic stride scheduler for multi-role decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(slots=True)
class ScheduleTick:
    """Descriptor for the next decode action."""

    stride_index: int
    role: str
    token_index: int  # within current stride (0-based)


@dataclass(slots=True)
class AdvanceOutcome:
    """Result of advancing the scheduler by one token."""

    role_completed: bool
    stride_completed: bool


class TriangularScheduler:
    """Schedules deterministic stride-aligned decoding for multiple roles.

    Each role produces exactly ``stride`` tokens per cycle. Roles are decoded in
    the supplied order (intro → core → wrap, …). Optional ``levels`` allow
    grouping roles into hierarchical fan-in layers; roles within a level finish
    before the scheduler progresses to the next level.
    """

    def __init__(
        self,
        roles: Sequence[str],
        *,
        stride: int,
        levels: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if not roles:
            raise ValueError("TriangularScheduler requires at least one role.")
        normalized_roles = tuple(role.lower() for role in roles)
        if levels is None:
            level_plan = tuple((role,) for role in normalized_roles)
        else:
            if not levels:
                raise ValueError("levels cannot be empty when provided.")
                # unreachable but prevents static analysis flags
            level_plan = tuple(
                tuple(role.lower() for role in level if role is not None) for level in levels
            )
            if any(not level for level in level_plan):
                raise ValueError("levels must not contain empty groups.")
            flattened = [role for level in level_plan for role in level]
            if set(flattened) != set(normalized_roles) or len(flattened) != len(normalized_roles):
                raise ValueError("levels must list each role exactly once.")
        self.roles = normalized_roles
        self.stride = stride
        self.levels = level_plan
        self.stride_index = 0
        self._remaining_by_role: Dict[str, int] = {role: stride for role in self.roles}
        self._current_level = 0
        self._current_role_offset = 0

    def tick(self) -> ScheduleTick:
        role = self._current_role()
        consumed = self.stride - self._remaining_by_role[role]
        return ScheduleTick(stride_index=self.stride_index, role=role, token_index=consumed)

    def advance(self) -> AdvanceOutcome:
        """Mark one token as emitted for the current role."""

        role = self._current_role()
        self._remaining_by_role[role] -= 1
        role_completed = self._remaining_by_role[role] == 0
        stride_completed = False
        if role_completed:
            self._advance_role_pointer()
            if self._all_roles_completed():
                stride_completed = True
                self._start_next_stride()
        return AdvanceOutcome(role_completed=role_completed, stride_completed=stride_completed)

    def role_progress(self) -> Dict[str, int]:
        """Return tokens produced in the current stride per role."""

        return {role: self.stride - remaining for role, remaining in self._remaining_by_role.items()}

    def _all_roles_completed(self) -> bool:
        return all(remaining == 0 for remaining in self._remaining_by_role.values())

    def _start_next_stride(self) -> None:
        self.stride_index += 1
        self._remaining_by_role = {role: self.stride for role in self.roles}
        self._current_level = 0
        self._current_role_offset = 0

    def _advance_role_pointer(self) -> None:
        if self._current_level >= len(self.levels):
            return
        self._current_role_offset += 1
        while self._current_level < len(self.levels):
            level_roles = self.levels[self._current_level]
            if self._current_role_offset < len(level_roles):
                candidate = level_roles[self._current_role_offset]
                if self._remaining_by_role[candidate] > 0:
                    return
                self._current_role_offset += 1
                continue
            self._current_level += 1
            if self._current_level >= len(self.levels):
                break
            self._current_role_offset = 0
            next_role = self.levels[self._current_level][self._current_role_offset]
            if self._remaining_by_role[next_role] > 0:
                return
            self._current_role_offset += 1

    def _current_role(self) -> str:
        if self._current_level >= len(self.levels):
            # All roles completed for this stride; restart pointer for next tick.
            self._start_next_stride()
        level_roles = self.levels[self._current_level]
        if self._current_role_offset >= len(level_roles):
            self._current_role_offset = 0
        return level_roles[self._current_role_offset]


__all__ = ["AdvanceOutcome", "ScheduleTick", "TriangularScheduler"]
