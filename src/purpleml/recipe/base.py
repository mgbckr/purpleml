import logging, warnings
import functools
from collections import OrderedDict



class Recipe():
        
    NOOP = '<NOOP>'
        
    def __init__(self, cache:bool=True):
        
        self.steps = OrderedDict()
        
        self.latest_steps = OrderedDict()
        self.last_step = None
        
        if type(cache) == bool:
            if cache:
                self.cache = dict()
            else:
                self.cache = None
        else:
            self.cache = cache
                
    
    def step(self, step_name:str=None, option_name=None, step_type=None, default_option: str|bool=False):
        
        if step_type not in ["branch", "optional branch", "switch", "fixed", None]:
            raise ValueError(f"Unknown step type: {step_type}")
        
        def decorator_recipe_ingredient(func):

            @functools.wraps(func)
            def wrapper_recipe_ingredient(*args, **kwargs):
                value = func(*args, **kwargs)
                return value

            function_name = func.__name__
            
            # figure out naming scheme
            if step_name is None and option_name is None:
                split = function_name.split("__")
                if len(split) == 1:
                    step_name_final = option_name_final = split[0]
                elif len(split) == 2:
                    step_name_final, option_name_final = split
                else:
                    raise ValueError(f"Unknown function name format: {function_name}")
            else:
                if step_name is None:
                    step_name_final = function_name
                else:
                    step_name_final = step_name

                if option_name is None:
                    option_name_final = function_name
                else:
                    option_name_final = option_name
            
            # print(f"Step name: {step_name}, option name: {option_name}")
            # print(f"Step name: {step_name_final}, option name: {option_name_final}")
            
            # get step information or initialize step
            if step_name_final in self.steps and self.steps[step_name_final] is not None:
                print(f"Check step: Step exists: {step_name_final}")
                step_dict = self.steps[step_name_final]
                if step_type is not None and step_dict['meta']['step_type'] != step_type:
                    raise ValueError(f"Step type does not match. Current: {step_dict['meta']['step_type']}; New: {step_type}")                    
            else:
                print(f"Check step: Creating new step: {step_name_final}")
                if step_type is None:
                    raise ValueError(f"Step type must not be `None`.")
                step_dict = {
                    "meta": {"step_type": step_type}, # default_option and latest_option are set below
                    "options": OrderedDict()
                }
                
                # add optional
                if step_type in ["optional branch", "switch"]:
                    step_dict["options"][Recipe.NOOP] = lambda x: x 
                    print(f"Add option: {Recipe.NOOP}")  
                
                self.steps[step_name_final] = step_dict

            # set step
            if option_name_final in step_dict:
                warnings.warn(f"Check function: Function exists. Overwriting option: {option_name_final}")
            step_dict["options"][option_name_final] = func
            step_dict["meta"]["latest_option"] = option_name_final
            
            if type(default_option) == bool:
                if default_option:
                    step_dict["meta"]["default_option"] = option_name_final
                else:
                    step_dict["meta"]["default_option"] = None
            elif type(default_option) == str:
                if default_option not in step_dict["options"]:
                    raise ValueError(f"Unknown default option: {default_option}")
                else:
                    step_dict["meta"]["default_option"] = default_option
                    step_dict["meta"]["latest_option"] = default_option
            else:
                raise ValueError(f"Unknown default option format: {default_option}")
                
            print(f"Add option: {option_name_final}")
            
            # set latest
            self.latest_steps[step_name_final] = {option_name_final: func}
            self.last_step = step_name_final             
                
            return wrapper_recipe_ingredient

        return decorator_recipe_ingredient


    def _paths(
            self, 
            prefix: list[((str, str), callable)], 
            steps: list[(str, dict)], 
            last_step: str=None,
            mode="all",
            config=None) -> list[((str, str), callable)]:
        
        if last_step is not None and last_step not in self.steps:
            raise ValueError(f"Last step does not exist: {last_step}")
        
        config = Recipe._check_config(config)
        
        # print("////")
        # print("Prefix:", prefix)
        # print("Steps:", steps)
        
        if (len(steps) == 0) or (len(prefix) > 0 and prefix[-1][0][0] == last_step):
            yield prefix

        else:
            
            step_name, step_data = steps[0]
            # print(steps[0], step_name, step_data)
            
            # get options
            if config is not None and step_name in config:
                options = [(k,v) for k,v in step_data["options"].items() if k in config[step_name]]
            else:
                if mode == "all":
                    if step_data["meta"]["default_option"] is not None:
                        default_option = step_data["meta"]["default_option"]
                        options = [(default_option, step_data["options"][default_option])]
                    else:
                        options = list(step_data["options"].items())
                elif mode == "latest":
                    latest_option = step_data["meta"]["latest_option"]
                    # print("Latest option:", latest_option)
                    options = [(latest_option, step_data["options"][latest_option])]
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            
            for option_name, option_func in options:
                yield from self._paths(
                    prefix + [((step_name, option_name), option_func)], 
                    steps[1:],
                    last_step=last_step,
                    mode=mode,
                    config=config)

    def _check_config(config):
        if config is None:
            return None
        config = config.copy()
        for k, v in config.items():
            if type(v) == str:
                config[k] = [v]
        return config
                
    def _format_path(self, path: list[((str, str), callable)]):
        return "__".join([f"{k}={v}" for (k,v), _ in path])
    
    
    def _execute(
            self, 
            steps: OrderedDict[str, dict], 
            input_data, 
            last_step: str=None, 
            format_path: bool=True,
            mode: str="all",
            config=None,
            loop_func=None):
        
        # print("Path Mode:", mode)
        paths = list(self._paths([], list(steps.items()), last_step=last_step, mode=mode, config=config))
        # print("Paths:", paths)
        
        results = []
        path_loop = loop_func(paths) if loop_func is not None else paths
        for path in path_loop:
            
            # formatted path
            formatted_path = self._format_path(path)
            
            if formatted_path in self.cache:
                print(f"Path: Retrieve cached: {formatted_path}")
                intermediate = self.cache[formatted_path]
                
            else:
                print(f"Path: Execute functions: {formatted_path}")
                functions = [f for _, f in path]
                intermediate = input_data
                for p, func in path:
                    intermediate = func(intermediate)
                
                # cache that
                self.cache[formatted_path] = intermediate
            
            # add to results
            results.append((formatted_path if format_path else path, intermediate))
            
        return results
       
        
    def clean_cache(self) -> bool:
        """Clean cache based on step additions"""
        if self.cache is not None:
            self.cache.clear()
            return True
        else:
            return False

    
    def reset_step(self, step:str):
        self.steps[step] = None
        
        
    def execute(self, input_data=None, last_step: str=None, format_path: bool=True, mode='all', config=None, loop_func=None):
        return self._execute(
            self.steps, 
            input_data=input_data, 
            last_step=last_step,
            format_path=format_path,
            mode=mode,
            config=config,
            loop_func=loop_func)

    
    def execute_latest(self, input_data, last_step: str=None, format_path: bool=True, result_only=True, config=None):
        
        if result_only:
            return self._execute(
                self.steps, 
                input_data=input_data, 
                format_path=format_path, 
                last_step=self.last_step if last_step is None else last_step,
                mode="latest",
                config=config)[0][1]
        
        else:
            return self._execute(
                self.steps, 
                input_data=input_data, 
                format_path=format_path, 
                last_step=self.last_step if last_step is None else last_step,
                mode="latest",
                config=config)[0]
        
    def __str__(self):
        string = "Recipe:\n"
        for step_name, step_data in self.steps.items():
            string += f"  - {step_name}\n"
            string += f"    - type:           {step_data['meta']['step_type']}\n"
            string += f"    - default_option: {step_data['meta']['default_option']}\n"
            string += f"    - latest_option:  {step_data['meta']['latest_option']}\n"
            string += f"    - options:\n"
            for option_name, _ in step_data["options"].items():
                string += f"      - {option_name}\n"
        return string
