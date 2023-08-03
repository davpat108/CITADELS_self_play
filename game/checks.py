from game.game import Game
from copy import deepcopy


def recursive_memory_check(obj1, obj2):
    # Base cases
    if id(obj1) == id(obj2):
        if obj1 is None:
            return False
        if isinstance(obj1, (int, float, complex, bool, str, tuple, range, frozenset, bytes)):
            return False
        return True

    if isinstance(obj1, list) and isinstance(obj2, list):
        for i in obj1:
            for j in obj2:
                if recursive_memory_check(i, j):
                    print(f"Attribute '{i}' in {
                        
                    } in first class and attribute '{j}' in {obj2} in second class are referencing the same memory.")
                    return True
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        for i in obj1.values():
            for j in obj2.values():
                if recursive_memory_check(i, j):
                    print(f"Attribute '{i}' in {obj1} in first class and attribute '{j}' in {obj2} in second class are referencing the same memory.")
                    return True
    elif hasattr(obj1, '__dict__') and hasattr(obj2, '__dict__'):
        return recursive_memory_check(vars(obj1), vars(obj2))

    return False


def check_same_memory_address(class_instance_1, class_instance_2):
    attrs_1 = vars(class_instance_1)
    attrs_2 = vars(class_instance_2)

    for attr1, value1 in attrs_1.items():
        for attr2, value2 in attrs_2.items():
            if recursive_memory_check(value1, value2):
                print(f"Attribute '{attr1}' in first class and attribute '{attr2}' in second class are referencing the same memory.")
                return True
    print("No attributes are referencing the same memory.")
    return False
