import itertools
import logging
import itertools
import random

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

def standard_ranking(candidates):
    def avg_score(candidate):
        prompt, code, evals = candidate
        score = sum(e.score() for e in evals) / len(evals)
        return score

    return sorted(candidates, key=avg_score, reverse=True)

def lexicase_ranking(candidates):
    pool = [evals for prompt, code, evals in candidates]

    case_count = min(len(evals) for evals in pool)
    cases = list(range(case_count))
    random.shuffle(cases)

    for case_order in itertools.combinations(cases, case_count):
        logging.info(f"Lexicase: test case order {reversed(case_order)}")
        # Pseudocode from page 3 of
        # Spector 2012 "Assessment of problem modality by differential performance of lexicase selection in genetic programming: a preliminary report"
        round_winners = range(len(pool))

        # Loop until a single candidate is left
        # Reversing case_order ensures diversity: the first case is always different
        for case in reversed(case_order):
            fitnesses = [pool[idx][case].score() for idx in round_winners]
            logging.info(f"Lexicase: "
                         f"idx: fitness values"
                         f"(test pass rates of all candidate programs on test {case})"
                         f"{[':'.join([str(idx), str(fitness)]) for idx, fitness in zip(round_winners, fitnesses)]}")
            best_fitness = max(fitnesses)

            round_winners = [idx for idx, fitness 
                             in zip(round_winners, fitnesses) 
                             if fitness == best_fitness]

            logging.info(f"Lexicase: "
                         f"programs that have max test pass rate of value {best_fitness} on test {case}) {round_winners}")
            
            if len(round_winners) == 1:
                break

        for idx in round_winners:
            yield candidates[idx]
        for idx in round_winners:
            del candidates[idx]

def beam_search(beam, update, ranking=standard_ranking, beam_width=100):
    """Generic evolutionary algorithm for improving anything"""
    new_beam = []

    # yield beam_width draft (non-repaired) programs
    for candidate in beam:
        yield candidate
        new_beam.append(candidate)

    while True:
        beam = itertools.islice(ranking(new_beam), beam_width)
        new_beam = []

        # yield beam_width * branching_factor children (repaired programs)
        for parent in beam:
            for child in update(parent):
                yield child
                new_beam.append(child)

        if len(new_beam) == 0:
            break

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


def draft(task_description, start_prompt, batch_size=10, limit_n=None,
          log_gpt_call=lambda **kwargs: print(kwargs)):
    t, delta_t = distribute_heat(1, limit_n, batch_size)
        
    codes = explore_gpt(source=start_prompt, instruction=task_description, modality='code',
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

def print_code(code, **vars):
    print(vars)
    print(code)

def develop(task_description,
            start_prompt,
            critics,
            language='C++',
            beam_width=3,
            branching_factor=10,
            lexicase=False,
            max_programs=None,
            log_metrics=print,
            log_attempt=print_code,
            log_solution=lambda *args, **kwargs: print('This program is the best!'),
            log_gpt_call=lambda *args, **kwargs: print(kwargs),
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

    def have_kids(candidate):
        logging.debug(f'Running debug_and_test')
        prompt, code, evals = candidate
        worst_eval = min(evals, key=lambda e: e.score())
        feedback = worst_eval.pen_report()
        
        for code in debug(code, feedback,
                          n=branching_factor, batch_size=batch_size,
                          log_gpt_call=log_gpt_call):
            yield feedback, code, [critic(code) for critic in critics]
    
    beam = draft(task_description, start_prompt, batch_size=batch_size, 
                 limit_n=beam_width, log_gpt_call=log_gpt_call)
    beam = ((task_description, code, [critic(code) for critic in critics])
            for code in beam)
    
    best_score = float('-inf')

    ranking = lexicase_ranking if lexicase else standard_ranking
    search = beam_search(beam, have_kids, ranking, beam_width)
    for idx, candidate in enumerate(search):
        prompt, code, evals = candidate

        avg_score = sum(e.score() for e in evals) / len(evals)
        test_pass_rate = sum(e.check() for e in evals) / len(evals)

        logging.info(f'Current program:\n{code}')

        metrics = {
            'idx': idx,
            'avg_score': avg_score,
            'pass_rate': test_pass_rate
        }

        log_metrics(metrics)
        log_attempt(code, idx=idx, 
                    prompt=prompt, test_pass_rate=test_pass_rate)

        if avg_score > best_score:
            best_score = avg_score
            log_metrics({f'best_{metric}': val for metric, val in metrics.items()})
            log_solution(code, idx=idx, 
                         prompt=prompt, test_pass_rate=test_pass_rate)

            if test_pass_rate == 1:
                break

        if max_programs is not None and (idx == max_programs - 1):
            break

    return code


if __name__ == '__main__':
    from seidr.eval import IOMatch

    for language in ('Python', 'C++'):
        # Use the same IO examples for prompt and tests
        critics = [
            lambda code: IOMatch(code, language=language, 
                                input=[''], output=['Hello World'])
        ]
        develop(task_description=f'A {language} program that outputs "Hello World"', 
                start_prompt=None,
                critics=critics, 
                language=language)
