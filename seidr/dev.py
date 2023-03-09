import itertools
import logging

from programlib import Program

from seidr.gpt import explore_gpt
from seidr.prompt import initial_prompt, write_debug_prompt, start_coding


def rolling_best(objects, max_score=1, metric=lambda x: x):
    best_score = None

    for object in objects:
        score = metric(object)
        if best_score is None or score > best_score:
            best_score = score
            try:
                logging.info(f'\nThe program has improved. Code: \n\n{object.read()}\n\n')
            except:
                pass
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
        if len(beam) == 0:
            break
        new_beam = []

        for parent in beam:
            for child in update(parent):
                yield child
                new_beam.append(child)


def distribute_heat(heat, n, batch_size):
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


def draft(task_description, examples, language, batch_size=10, limit_n=None, 
          log_gpt_call=lambda **kwargs: print(kwargs)):
    t, delta_t = distribute_heat(1, limit_n, batch_size)
        
    prompt = initial_prompt(task_description, examples)
    start = start_coding(prompt, language=language)
    codes = explore_gpt(source=start, instruction=task_description, modality='code',
                        batch_size=batch_size, 
                        t=t, 
                        delta_t=delta_t,
                        log_gpt_call=log_gpt_call)

    if limit_n:
        codes = itertools.islice(codes, limit_n)

    return codes


def debug(code, debug_prompt_text, n, batch_size=10, log_gpt_call=print):
    """Generate n attempts to fix program so that it passes tests"""
    t, delta_t = distribute_heat(1, n, batch_size)

    codegen = explore_gpt(source=code,
                          instruction=debug_prompt_text,
                          modality='code',
                          batch_size=batch_size,
                          t=t, delta_t=delta_t,
                          log_gpt_call=log_gpt_call)
    return itertools.islice(codegen, n)

def pbe_critic(task_description, tests, debug_template='Make sure {i} -> {o}'):
    def critic(program):
        test_runs = program.test(tests)
        dp = write_debug_prompt(test_runs, debug_template, task_description)
        return program, dp
    return critic

def develop(task_description, 
            critic,
            examples,
            language='C++',
            beam_width=100,
            branching_factor=10,
            max_programs=None,
            log_metrics=print,
            log_program=lambda p: print(p.read()),
            log_gpt_call=lambda **kwargs: print(kwargs),
            batch_size=10):
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
    codes = draft(task_description, examples, language, batch_size=batch_size, 
                  limit_n=beam_width, log_gpt_call=log_gpt_call)

    beam = (critic(Program(code, language=language)) for code in codes)

    def debug_and_test(candidate):
        logging.debug(f'Running debug_and_test')
        program, debug_prompt = candidate
        
        for code in debug(program.read(), debug_prompt,
                          n=branching_factor, batch_size=batch_size,
                          log_gpt_call=log_gpt_call):
            yield critic(Program(code, language=language))

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
            logging.info(f'\nProgram idx: {idx}\n')
            if int(idx) % 10 == 0:
                logging.info(f'\nProgram code:\n{program.read()}')
            yield program

            if max_programs and idx >= max_programs:
                break
    solutionogen = beam_search(beam, debug_and_test, lambda candidate: candidate[0].avg_score, beam_width)
    solutionogen = (program for program, instruction in solutionogen)
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
    critic = pbe_critic(task, examples)
    develop(task, critic, examples)
