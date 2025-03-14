import itertools
import logging
from programlib import Program, Language
from typing import Callable, Optional, Iterable, Tuple, List, Generator
import random
import time

from seidr.llm import explore_llm, default_batch_size
from seidr.eval import Evaluation


def standard_ranking(
        candidates: List[Tuple[str, str, list[Callable[[Program], Evaluation]]]],
) -> List[Tuple[str, str, list[Callable[[Program], Evaluation]]]]:
    def avg_score(
            candidate: Tuple[str, str, list[Callable[[Program], Evaluation]]],
    ) -> float:
        prompt, code, evals = candidate
        score = sum(e.score() for e in evals) / len(evals)
        return score

    return sorted(candidates, key=avg_score, reverse=True)


def lexicase_ranking(
        candidates: List[Tuple[str, str, list[Callable[[Program], Evaluation]]]],
) -> Tuple[str, str, list[Callable[[Program], Evaluation]]]:
    pool = [evals for prompt, code, evals in candidates]

    case_count = min(len(evals) for evals in pool)
    cases = list(range(case_count))
    random.shuffle(cases)

    for case_order in itertools.combinations(cases, case_count):
        logging.info(f"Lexicase: test case order {list(reversed(case_order))}")
        # Pseudocode from page 3 of
        # Spector 2012 "Assessment of problem modality by differential performance of lexicase selection in genetic programming: a preliminary report"
        round_winners = range(len(pool))

        # Loop until a single candidate is left
        # Reversing case_order ensures diversity: the first case is always different
        for case in reversed(case_order):
            fitnesses = [pool[idx][case].score() for idx in round_winners]
            logging.info(
                f"Lexicase: "
                f"idx: fitness values on test number {case}"
                f"{[': '.join([str(idx), str(fitness)]) for idx, fitness in zip(round_winners, fitnesses)]}"
            )
            best_fitness = max(fitnesses)

            round_winners = [
                idx
                for idx, fitness in zip(round_winners, fitnesses)
                if fitness == best_fitness
            ]

            logging.info(
                f"Lexicase: "
                f"programs that have max test pass rate of value {best_fitness} on test {case}) {round_winners}"
            )

            if len(round_winners) == 1:
                break

        for idx in round_winners:
            yield candidates[idx]
        for idx in round_winners:
            del candidates[idx]


def beam_search(
        beam: List | Generator,
        update: Callable,
        ranking: Callable = standard_ranking,
        beam_width: int = 100,
) -> Generator:
    """Generic evolutionary algorithm for improving anything"""
    while True:
        parents = []
        for code in beam:
            yield code
            parents.append(code)

        if len(parents) == 0:
            break

        parents = itertools.islice(ranking(parents), beam_width)
        beam = (child for parent in parents for child in update(parent))


def distribute_heat(heat: float, n: int, batch_size: int) -> Tuple[float, float]:
    """Calculate the start temperature `t` and steps `delta_t`
    for increasing temperature of an LLM that will be distributed over the batch."""
    # Setting: we generate `n` outputs in total from an LLM by generating `batch_size` outputs at once,
    # Note that we use `batch_size`=1 for langchain models
    if n == 1:
        t = 0
        delta_t = 0.2
    else:
        # We intentionally avoid temperature=0
        # That would lead to a batch of identical code snippets
        # Update temperature but keep it 1 at max
        batch_count = n // batch_size + 1
        delta_t = heat / batch_count
        t = delta_t

    return t, delta_t


def print_code(code, **vars):
    print(vars)
    print(code)


class SEIDR:
    """Generate, test and debug programs for a set of given problem descriptions and tests"""

    def __init__(
            self,
            task_name: str,
            task_description: str,
            critics: list[Callable[[Program], Evaluation]],
            model_name: str,
            language: str | Language,
            beam_width: int = 10,
            drafts_per_prompt: int = 10,
            explanations_per_program: int = 10,
            repairs_per_explanation: int = 2,
            lexicase_selection: bool = False,
            log_metrics: Callable = print,
            log_attempt: Callable = print_code,
            log_solution: Callable = lambda *args, **kwargs: print('This program is the best!'),
            log_llm_call: Callable = lambda **kwargs: print(kwargs),
            max_programs: Optional[int] = None,
            batch_size: Optional[int] = None,
            ollama_url: Optional[str] = None,
            delay: int = 0  # for Anthropic models
    ) -> None:
        self.task_name = task_name
        self.task_description = task_description
        self.critics = critics
        self.model_name = model_name
        self.language = language
        self.beam_width = beam_width
        self.drafts_per_prompt = drafts_per_prompt
        self.explanations_per_program = explanations_per_program
        self.repairs_per_explanation = repairs_per_explanation
        self.lexicase_selection = lexicase_selection
        self.log_metrics = log_metrics
        self.log_attempt = log_attempt
        self.log_solution = log_solution
        self.log_llm_call = log_llm_call
        self.max_programs = max_programs
        self.ollama_url = ollama_url
        self.delay = delay

        if not batch_size:
            batch_size = default_batch_size(model_name)
        self.batch_size = batch_size

    def draft(self, start_code: str = "") -> Iterable[str]:
        """Create a draft solution with the "generate" prompt template
        that requires first code lines in `start_code`, task description and programming language"""
        batch_size = min(self.batch_size, self.drafts_per_prompt)
        t, delta_t = distribute_heat(1, self.drafts_per_prompt, batch_size)

        return itertools.islice(
            explore_llm(
                t=t,
                delta_t=delta_t,
                mode="generate",
                model_name=self.model_name,
                language=self.language,
                task_name=self.task_name,
                task_description=self.task_description,
                start_code=start_code,
                log_llm_call=self.log_llm_call,
                batch_size=batch_size,
                base_url=self.ollama_url,
            ),
            self.drafts_per_prompt,
        )

    def repair(self, code: str, feedback: str) -> Iterable[str]:
        """Generate `self.explanations_per_program` * `self.repairs_per_explanation` attempts
        to fix the program in `code` so that it passes tests"""
        explain_batch_size = min(self.batch_size, self.explanations_per_program)
        repair_batch_size = min(self.batch_size, self.repairs_per_explanation)

        explain_t, explain_delta_t = distribute_heat(
            1, self.explanations_per_program, self.batch_size
        )
        repair_t, repair_delta_t = distribute_heat(
            1, self.repairs_per_explanation, self.batch_size
        )

        for bug_summary in itertools.islice(
                explore_llm(
                    t=explain_t,
                    delta_t=explain_delta_t,
                    mode="explain_bugs",
                    model_name=self.model_name,
                    language=self.language,
                    task_name=self.task_name,
                    task_description=self.task_description,
                    code=code,
                    issue=feedback,
                    log_llm_call=self.log_llm_call,
                    batch_size=explain_batch_size,
                    base_url=self.ollama_url,
                ),
                self.explanations_per_program,
        ):
            for repair in itertools.islice(
                    explore_llm(
                        t=repair_t,
                        delta_t=repair_delta_t,
                        mode="repair",
                        model_name=self.model_name,
                        language=self.language,
                        task_name=self.task_name,
                        task_description=self.task_description,
                        input=input,
                        code=code,
                        bug_summary=bug_summary,
                        log_llm_call=self.log_llm_call,
                        batch_size=repair_batch_size,
                        base_url=self.ollama_url,
                    ),
                    self.repairs_per_explanation,
            ):
                yield repair

    def develop(self, start_code: str = "") -> str:
        """
        Write a program in a programming language that solves task and passes tests.
        Solve repair-rewrite trade-off with beam search of given beam size

        examples is a sequence of (inputs, outputs) pairs
        where inputs and outputs are sequences of strings (lines of code)
        likewise for tests

        examples are used in the prompt for the language model,
        while tests are used to select the best solution

        Returns a generator of programs where each program passes
        more tests than the previous one. The last program in the generator
        passes all tests.
        """

        def have_kids(
                candidate: Tuple[str, str, list[Callable[[Program], Evaluation]]],
        ) -> Tuple[str, str, list[Callable[[Program], Evaluation]]]:
            prompt, code, evals = candidate
            worst_eval = min(evals, key=lambda e: e.score())
            feedback = worst_eval.pen_report()

            for code in self.repair(code, feedback):
                yield feedback, code, [critic(code) for critic in self.critics]

        drafts = self.draft(start_code)
        drafts = (
            (self.task_description, code, [critic(code) for critic in self.critics])
            for code in drafts
        )

        best_score = float("-inf")
        best_code = None

        ranking = lexicase_ranking if self.lexicase_selection else standard_ranking
        search = beam_search(
            beam=drafts, update=have_kids, ranking=ranking, beam_width=self.beam_width
        )

        for idx, candidate in enumerate(search):
            prompt, code, evals = candidate

            avg_score = sum(e.score() for e in evals) / len(evals)
            test_pass_rate = sum(e.check() for e in evals) / len(evals)

            logging.info(f"Prompt:\n{prompt}\n")
            logging.info(f"The program generated with the prompt above:\n{code}")

            metrics = {"idx": idx, "avg_score": avg_score, "pass_rate": test_pass_rate}

            self.log_metrics(metrics)
            self.log_attempt(
                code, idx=idx, prompt=prompt, test_pass_rate=test_pass_rate
            )

            if avg_score > best_score:
                best_score = avg_score
                best_code = code
                self.log_metrics(
                    {f"best_{metric}": val for metric, val in metrics.items()}
                )
                logging.info(
                    "\n".join(
                        [f"\nbest_{metric}: {val}\n" for metric, val in metrics.items()]
                    )
                )
                self.log_solution(
                    code, idx=idx, prompt=prompt, test_pass_rate=test_pass_rate
                )

                if test_pass_rate == 1:
                    break

            if self.max_programs is not None and (idx == self.max_programs - 1):
                break

            time.sleep(self.delay)

        return best_code
