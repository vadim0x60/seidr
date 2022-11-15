from gpt import explore_gpt
from prompt import initial_prompt, debug_prompt, start_coding
import functools
import itertools
from programlib import Program

def rolling_best(candidates, log_f, max_score=1):
    best_score = float('-inf')
    
    for program in candidates:
        best_program = None

        if program.score > best_score:
            best_score = program.score
            best_program = program

        log_f({
            'score': program.score,
            'best_score': best_score,
        })

        if best_program:
            yield best_program

            if best_program.score == max_score:
                break        

def beam_search(beam, update, metric, beam_width=100):
    """Generic evolutionary algorithm for improving anything"""

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
    for code in explore_gpt(start, batch_size=batch_size):
        program = Program(code, language=language)
        program.test(tests)
        yield program

def debug(program, tests, language, n, batch_size=10):
    """Generate n attempts to fix program so that it passes tests"""
    
    assert program.score != 1

    codogen = explore_gpt(program.read(), 
                          instruction=debug_prompt(program),
                          batch_size=batch_size,
                          heat_per_batch=batch_size / n)

    for code in itertools.islice(codogen, n):
        program = Program(code, language)
        program.test(tests)
        yield program

def develop(task, task_description, tests, language='C++', 
            beam_size=100, branching_factor=100, log_f=lambda x: x,
            batch_size=10):
    """
    Write a program in language that solves task and passes tests.
    Solve debug-rewrite trade-off with beam search of given beam size

    https://vadim.me/publications/unreasonable#search
    """

    update = functools.partial(debug, tests=tests, language=language, 
                               n=branching_factor, batch_size=batch_size)
    metric = lambda program: program.score
                               
    beam = draft(task, task_description, tests, language, batch_size=batch_size)
    if beam_size:
        beam = itertools.islice(beam, beam_size)
    solutionogen = beam_search(beam, update, metric, beam_size)

    return rolling_best(solutionogen, log_f)

if __name__ == '__main__':
    tests = [([], ['Hello World'])]
    *_, perfect_solution = develop('Hello World', 'Write a program that prints "Hello World"', tests, log_f=print, language='C++')
    print(perfect_solution.read())