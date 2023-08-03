from purpleml.recipe.base import Recipe


def test_basic():
    """Not really testing anything yet."""

    # TEST
    recipe = Recipe()

    string = "BLUBB"

    @recipe.step(step_type="fixed")
    def first(input_str):
        return input_str + "_first"

    @recipe.step("pp", step_type="branch")
    def test1(input_str):
        return input_str + "_test1"

    @recipe.step("pp")
    def test2(input_str):
        return input_str + "_test2"

    print()
    print(str(recipe))

    recipe.execute("blubb", last_step="first")
    # recipe.execute_latest("blubb")