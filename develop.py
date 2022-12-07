from gpt import explore_gpt
from prompt import initial_prompt, debug_prompt, start_coding
import functools
import itertools
from programlib import Program

def rolling_best(objects, max_score=1, metric = lambda x: x):
    best_score = None

    for object in objects:
        score = metric(object)
        if best_score is None or score > best_score:
            best_score = score
            yield object

        if best_score >= max_score:
            break

def beam_search(beam, update, metric, beam_width=100):
    """Generic evolutionary algorithm for improving anything"""

    new_beam = []

    for code in beam:
        yield code
        new_beam.append(code)

    while True:
        beam = sorted(new_beam, key=metric, reverse=True)[:beam_width]
        new_beam = []

        for parent in beam:
            for child in update(parent):
                yield child
                new_beam.append(child)

def distribute_heat(heat, n, batch_size):
    batch_count = n // batch_size + 1
    heat_per_batch = heat / batch_count
    return heat_per_batch

def draft(task_description, examples, language, batch_size=10, limit_n=None):
    heat_per_batch = distribute_heat(1, limit_n, batch_size) if limit_n else 0.2
    prompt = initial_prompt(task_description, examples)
    start = start_coding(prompt, language=language)

    codes = explore_gpt(start, batch_size=batch_size, 
                               heat_per_batch=heat_per_batch)

    if limit_n:
        codes = itertools.islice(codes, limit_n)

    return codes

def debug(code, test_runs, n, batch_size=10):
    """Generate n attempts to fix program so that it passes tests"""

    return explore_gpt(code, 
                       instruction=debug_prompt(test_runs),
                       batch_size=batch_size,
                       heat_per_batch=distribute_heat(1, n, batch_size))

def test(code, tests, language='C++'):
    program = Program(code, language=language)
    return program, program.test(tests)

def develop(task_description, examples=tuple(), tests=tuple(), language='C++', 
            beam_size=100, branching_factor=10, 
            log_metrics=print, log_program=lambda p: print(p.read()),
            batch_size=10, max_programs=None):
    """
    Write a program in language that solves task and passes tests.
    Solve debug-rewrite trade-off with beam search of given beam size

    examples is a sequence of (inputs, outputs) pairs
    where inputs and outputs are sequences of strings (lines of code)
    likewise for tests

    examples are used in the prompt for the language model,
    while tests are used to select the best solution

    Returns a generator of programs where each program passes
    more tests than the previous one. The last program in the generator
    passes all tests.
    """
    codes = draft(task_description, examples, language, 
                  batch_size=batch_size, limit_n=beam_size)    
    beam = (test(code, tests, language) for code in codes)

    def debug_and_test(candidate):
        program, test_runs = candidate

        for code in debug(program.read(), test_runs, branching_factor, batch_size=batch_size):
            yield test(code, tests, language)

    def metric_logger(prefix):
        def log(program):
            log_metrics({
                prefix + 'avg_score': program.avg_score,
                prefix + 'pass_rate': program.pass_rate,
            })

            return program
        return log

    def limit_n(programs):
        for idx, program in enumerate(programs):
            log_metrics({
                'idx': idx
            })

            yield program

            if max_programs and idx >= max_programs:
                break

    solutionogen = beam_search(beam, debug_and_test, lambda candidate: candidate[0].avg_score, beam_size)
    solutionogen = (program for program, test_runs in solutionogen)
    solutionogen = limit_n(solutionogen)
    solutionogen = map(metric_logger(''), solutionogen)
    solutionogen = rolling_best(solutionogen, max_score=1, metric=lambda prog: prog.avg_score)
    
    solution = None

    for solution in solutionogen:
        metric_logger('best_')(solution)
        log_program(solution)

    return solution

if __name__ == '__main__':
    task = 'A program that outputs "Hello World"'
    examples = [
        ([''], ['Hello World'])
    ]

    # Use the same IO examples for prompt and tests
    develop(task, examples, examples)