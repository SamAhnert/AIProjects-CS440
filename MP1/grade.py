#!/usr/bin/env python3
import pprint, argparse, pickle, json

import maze 

def fail(message):
    return {
        'score'     : 0,
        'output'    : message, 
        'visibility': 'visible', 
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description     = 'CS440 MP1 Autograder', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gradescope', default = False, action = 'store_true',
                        help = 'save output in gradescope-readable json file')

    arguments   = parser.parse_args()
    
    try:
        import search 
    except ImportError:
        message = 'could not find module \'search\', did you upload `search.py`?'
        if arguments.gradescope:
            with open('results.json', 'w') as file:
                file.write(json.dumps(fail(message)))
        else:
            print(message)
        raise SystemExit

def generate_answer_key(path, mazes, solutions):
    key_instructor  = tuple({case: getattr(search, solution)(maze) 
        for case, maze in mazes.items()}
        for mazes, solution in zip(mazes, solutions))
    key_student     = tuple({case: len(path) for case, path in part.items()} 
        for part in key_instructor)
    pickle.dump(key_instructor, open(path['instructor'], 'wb'))
    pickle.dump(key_student,    open(path['student'],    'wb'))

def load_answer_key(path):
    try:
        return pickle.load(open(path['instructor'], 'rb'))
    except FileNotFoundError:
        print('running in student mode (instructor key unavailable)')
        return pickle.load(open(path['student'],    'rb'))

def grade_closed(name, key, mazes, solution, weight = 1):
    def grade(case, maze):
        y = key[case]
        z = getattr(search, solution)(maze)
        # check that the path is valid 
        score_validity  = int(maze.validate_path(z) is None)
        # check that the length of the student’s path matches 
        score_length    = int(len(z) == (y if type(y) is int else len(y)))
        # if instructor key available, check that path vertices match  
        score_vertices  = int(type(y) is not int and z == y)
        
        return (
            {
                'name'      : '{0}: `validate_path(_:)` for \'{1}\' maze'.format(name, case),
                'score'     : 0.5 * weight * score_validity,
                'max_score' : 0.5 * weight,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: correct path length for \'{1}\' maze'.format(name, case),
                'score'     : 0.5 * weight * score_length,
                'max_score' : 0.5 * weight,
                'visibility': 'visible'
            },
            #{
            #    'name'      : '{0}: correct path vertices for \'{1}\' maze'.format(name, case),
            #    'score'     : 0.75 * weight * score_vertices,
            #    'max_score' : 0.75 * weight * (type(y) is not int),
            #    'visibility': 'visible'
            #},
        )
            
    return tuple(item for case, maze in mazes.items() for item in grade(case, maze))

def grade_open(name, key, mazes, solution, weight = 1):
    def grade(case, maze):
        y = key[case]
        z = getattr(search, solution)(maze)
        # check that the path is valid 
        score_validity  = int(maze.validate_path(z) is None)
        # score student path by dividing the length of an MST-based solution by 
        # the length of the student path 
        score_length    = min((y if type(y) is int else len(y)) / max(len(z), 1), 1)
        
        return (
            {
                'name'      : '{0}: `validate_path(_:)` for \'{1}\' maze'.format(name, case),
                'score'     : 0.5 * weight * score_validity,
                'max_score' : 0.5 * weight,
                'visibility': 'visible'
            },
            {
                'name'      : '{0}: path length for \'{1}\' maze'.format(name, case),
                'score'     : 0.5 * weight * score_length,
                'max_score' : 0.5 * weight,
                'visibility': 'visible'
            },
        )
            
    return tuple(item for case, maze in mazes.items() for item in grade(case, maze))

def main():    
    solutions = ('bfs', 'astar_single', 'astar_corner', 'astar_multiple', 'fast')
    for solution in solutions:
        if not hasattr(search, solution):
            return fail('module \'search\' is missing expected member \'{0}\''.format(solution))
        if not callable(getattr(search, solution)):
            return fail('member \'{0}\' in module \'search\' is not callable'.format(solution))
    
    mazes = (
        # part 1: 20 points total, 4 points per case
        {case: maze.maze('data/part-1/{0}'.format(case))
            for case in ('tiny', 'small', 'medium', 'large', 'open')},
        # part 2: 20 points total, 4 points per case 
        {case: maze.maze('data/part-2/{0}'.format(case))
            for case in ('tiny', 'small', 'medium', 'large', 'open')},
        # part 3: 30 points total, 10 points per case 
        {case: maze.maze('data/part-3/{0}'.format(case))
            for case in ('tiny', 'medium', 'large')},
        # part 4: 30 points total, 10 points per case 
        {case: maze.maze('data/part-4/{0}'.format(case))
            for case in ('tiny', 'small', 'medium')},
        # part 5: 22 points total, 22 points per case 
        #{case: maze.maze('data/part-5/{0}'.format(case))
        #    for case in ('large',)},
    )
    
    #generate_answer_key({'instructor': 'key', 'student': 'key-student'}, mazes, solutions)
    key             = load_answer_key({'instructor': 'key', 'student': 'key-student'})
    parts_closed    = tuple(item for i, points in zip(range(0, 4), (4, 4, 10, 10))
        for item in grade_closed('part-{0}'.format(i + 1), key[i], mazes[i], solutions[i], 
            weight = points))
    #parts_open      = tuple(item for i, points in zip(range(4, 5), (22,))
    #    for item in grade_open(  'part-{0}'.format(i + 1), key[i], mazes[i], solutions[i], 
    #        weight = points))
    
    # construct grade dictionary for gradescope 
    return {
        'visibility': 'visible', 
        'tests': parts_closed
        #'tests': parts_closed + parts_open
    } 
    
if __name__ == "__main__":
    results     = main()
    if arguments.gradescope:
        with open('results.json', 'w') as file:
            file.write(json.dumps(results))
    else:
        pprint.pprint(results)
    
