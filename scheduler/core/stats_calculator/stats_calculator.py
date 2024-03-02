# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter
from typing import Dict, Tuple, FrozenSet

from lucupy.minimodel import Band, ProgramID, NightIndex, Program
from lucupy.types import ZeroTime

from scheduler.core.components.collector import Collector
from scheduler.core.events_queue.night_changes import NightlyTimeline
from scheduler.core.plans import NightStats
from scheduler.graphql_mid.scalars import Sites
from scheduler.services import logger_factory


__all__ = [
    'StatCalculator',
]


logger = logger_factory.create_logger(__name__)


class StatCalculator:
    """
    Interface for stats in the calculation and results of plans.
    """

    @staticmethod
    def program_real_total_used(program: Program):
        return sum((o.part_time() + o.acq_overhead + o.prog_time() for o in program.observations()),
                   start=ZeroTime)

    @staticmethod
    def calculate_timeline_stats(timeline: NightlyTimeline,
                                 nights: FrozenSet[NightIndex],
                                 sites: Sites,
                                 collector: Collector) -> Dict[str, Tuple[str, float]]:

        scores_per_program: Dict[ProgramID, float] = {}
        programs = {}
        for night_idx in nights:
            for site in sites:
                for entry in timeline.timeline[night_idx][site]:
                    plan = entry.plan_generated  # Update last plan
                    n_toos = 0
                    plan_score = 0
                    plan_conditions = []
                    completion_fraction: Counter[Band] = Counter({b: 0 for b in Band})

                    for visit in plan.visits:
                        obs = collector.get_observation(visit.obs_id)
                        # check if obs is a too
                        if obs.too_type is not None:
                            n_toos += 1

                        # add to plan_score
                        plan_score += visit.score

                        # add used conditions
                        plan_conditions.append(obs.constraints.conditions)

                        # check completion
                        program = collector.get_program(obs.belongs_to)

                        scores_per_program.setdefault(program.id, 0)
                        scores_per_program[program.id] += visit.score
                        completion_fraction[program.band] += 1

                        # Calculate altitude data
                        ti = collector.get_target_info(visit.obs_id)
                        end_time_slot = visit.start_time_slot + visit.time_slots
                        values = ti[night_idx].alt[visit.start_time_slot: end_time_slot]
                        alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                        plan.alt_degs.append(alt_degs)

                    program_completion = {p.id: StatCalculator.calculate_program_completion(programs[p])
                                          for p in programs}
                    plan.night_stats = NightStats(f'{plan.time_left()} min',
                                                  plan_score,
                                                  n_toos,
                                                  completion_fraction,
                                                  program_completion)

        plans_summary = {}
        for p_id in scores_per_program:
            program = collector.get_program(p_id)
            total_used = program.total_used()
            prog_total = StatCalculator.program_real_total_used(program)

            completion = f'{float(total_used.total_seconds() / prog_total.total_seconds()) * 100:.1f}%'
            score = scores_per_program[p_id]
            # print(completion, score)
            plans_summary[p_id.id] = (completion, score)

        return plans_summary

    @staticmethod
    def calculate_program_completion(program: Program) -> str:
        total_used = program.total_used()
        prog_total = StatCalculator.program_real_total_used(program)
        return f'{float(total_used.total_seconds() / prog_total.total_seconds()) * 100:.1f}%'

