from functools import reduce
from barfi import Block                                                       
from dataclasses import dataclass                                             


def flat_map(f, xs):
    return reduce(lambda a, b: a + list(b), map(f, xs), [])


def flatten(xs):
    return flat_map(lambda x: x, xs)

def compose(*fs):
    def composed(x):
        for f in fs:
            x = f(x)
        return x

    return composed


@dataclass(frozen=True)
class Input:
    name: str

@dataclass(frozen=True)
class Output:
    name: str

def define_block(name, *args):
    def decorator(f):
        block = Block(name=name)
        inputs = tuple(filter(lambda a: isinstance(a, Input), args))
        outputs = tuple(filter(lambda a: isinstance(a, Output), args))
        for inp in inputs:
            block.add_input(inp.name)
        for out in outputs:
            block.add_output(out.name)

        def feed_func(self):
            ins = {
                inp.name: self.get_interface(inp.name)
                for inp in inputs                                             
            }
            outs = f(*ins.values())
            #outs = f(**ins)
            if len(outputs) == 1:
                self.set_interface(name=outputs[0].name, value=outs)
            elif len(outputs) > 1:
                for out, v in zip(outputs, outs):
                    self.set_interface(name=out.name, value=v)

        block.add_compute(feed_func)
        return block
    return decorator
