def my_decorator(func,func_hook):
    def wrapper(*args, **kwargs):
        func_hook(*args, **kwargs)
        result = func(**kwargs)
        return result
    return wrapper
        