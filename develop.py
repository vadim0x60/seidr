import itertools

import openai
from programlib import Program

from gpt import explore_gpt
from prompt import initial_prompt, debug_prompt, start_coding


def rolling_best(candidates, log_f, max_score=1):
    best_program = None

    for program, test_runs in candidates:
        if not best_program or program.avg_score > best_program.avg_score:
            best_program = program

        log_f({
            'best_avg_score': best_program.avg_score,
            'best_pass_rate': best_program.pass_rate,
            'avg_score': program.avg_score,
            'pass_rate': program.pass_rate,
        })

        if best_program == program:
            yield best_program

            if best_program.avg_score == max_score:
                break


def beam_search(beam, update, metric, beam_width=100):
    """Generic evolutionary algorithm for improving anything"""

    beam = list(beam)
    yield from beam

    while True:
        new_beam = []

        for parent in beam:
            for child in update(parent):
                yield child
                new_beam.append(child)

        beam = sorted(new_beam, key=metric, reverse=True)[:beam_width]


def draft(task, task_description, tests, language, batch_size=10):
    prompt = initial_prompt(task, task_description, tests)
    start = start_coding(prompt, language=language)
    return explore_gpt(start, batch_size=batch_size)


def debug(code, debug_prompt_text, test_runs, n, batch_size=10):
    """Generate n attempts to fix program so that it passes tests"""

    return explore_gpt(code,
                       instruction=debug_prompt(test_runs, debug_prompt_text),
                       batch_size=batch_size,
                       heat_per_batch=batch_size / n)


def test(code, tests, language='C++'):
    program = Program(code, language=language)
    return program, program.test(tests)


def develop(task, task_description, tests,
            debug_prompt_text='Make sure {i} -> {o}', language='C++',
            beam_size=100, branching_factor=100, log_f=lambda x: x,
            batch_size=10):
    """
    Write a program in language that solves task and passes tests.
    Solve debug-rewrite trade-off with beam search of given beam size

    https://vadim.me/publications/unreasonable#search
    """

    codes = draft(task, task_description, tests, language, batch_size=batch_size)
    beam = (test(code, tests, language) for code in codes)
    if beam_size:
        beam = itertools.islice(beam, beam_size)

    def debug_and_test(candidate):
        program, test_runs = candidate

        for code in debug(program.read(), debug_prompt_text, test_runs, branching_factor, batch_size=batch_size):
            yield test(code, tests, language)

    def success_metric(candidate):
        program, test_runs = candidate

        return program.avg_score

    solutionogen = beam_search(beam, debug_and_test, success_metric, beam_size)

    return rolling_best(solutionogen, log_f)


if __name__ == '__main__':
    openai.organization = "org-W4y3V2nef7qsGvILgzrMjzNW"
    tests = [([], ['Hello World'])]
    language = "C++"
    *_, perfect_solution = develop('Hello World', f'Write a program that prints "Hello World"',
                                   tests, log_f=print,
                                   language=language, beam_size=2, branching_factor=4, batch_size=8)
    print(perfect_solution.read())
