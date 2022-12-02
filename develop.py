from gpt import explore_gpt
from prompt import initial_prompt, debug_prompt, start_coding
import functools
import itertools
from programlib import Program

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

def draft(task, task_description, tests, language, batch_size=10, limit_n=None):
    heat_per_batch = distribute_heat(1, limit_n, batch_size) if limit_n else 0.2
    prompt = initial_prompt(task, task_description, tests)
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

def develop(task, task_description, tests, language='C++', 
            beam_size=100, branching_factor=100, log_f=lambda x: x,
            batch_size=10):
    """
    Write a program in language that solves task and passes tests.
    Solve debug-rewrite trade-off with beam search of given beam size

    tests parameter is a sequence of (inputs, outputs) pairs
    where inputs and outputs are sequences of strings (lines of code)

    Returns a generator of programs where each program passes
    more tests than the previous one. The last program in the generator
    passes all tests.
    """

    codes = draft(task, task_description, tests, language, 
                  batch_size=batch_size, limit_n=beam_size)    
    beam = (test(code, tests, language) for code in codes)

    def debug_and_test(candidate):
        program, test_runs = candidate

        for code in debug(program.read(), test_runs, branching_factor, batch_size=batch_size):
            yield test(code, tests, language)

    def success_metric(candidate):
        program, test_runs = candidate

        return program.avg_score

    solutionogen = beam_search(beam, debug_and_test, success_metric, beam_size)

    return rolling_best(solutionogen, log_f)

if __name__ == '__main__':
    tests = [([], ['Hello World'])]
    *_, perfect_solution = develop('Hello World', 'Write a program that prints "Hello World"', tests, log_f=print, language='C++')
    print(perfect_solution.read())