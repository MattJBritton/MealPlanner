# Imports
from typing import List
from IPython.display import display
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import altair as alt
import string
import ipywidgets as wid

# Global Var
current_meal_plan = []

# Load Data Function
def parse_data(raw_df):
    
    def clean_ingredient_names(col):
        return col\
        .str.replace(" ", "_")\
        .str.replace(",_", ",")\
        .str.replace('[^a-zA-Z,_]', '')\
        .str.replace('_{2,}', '_')\
        .str.strip()\
        .str.lower()

    # define needed columns
    ingredient_columns = {x: x.replace("Ingredients: ", "") for x in raw_df.columns if "Ingredients: " in x}

    # remove unnecessary columns, if any
    # data = data.loc[:,cols_to_keep]
    data = raw_df.rename(ingredient_columns, axis=1)
    ingredient_columns = list(ingredient_columns.values())
    other_columns = [x for x in data.columns if x != "Recipe Name" and x not in ingredient_columns]

    # clean data
    data = data.fillna("")
    for col in ingredient_columns:
        data[col] = clean_ingredient_names(data[col])

    ingredient_type_df = data.loc[:,ingredient_columns].melt(var_name = "Type", value_name = "Ingredients")\
    .query("Ingredients != ''")
    ingredient_type_df = ingredient_type_df["Type"].to_frame().join(
        ingredient_type_df["Ingredients"].str.split(',', expand=True)
    ).melt(id_vars="Type")
    ingredient_type_map = ingredient_type_df.drop("variable", axis=1).drop_duplicates().set_index("value")["Type"].to_dict()

    # make combined ingredients column
    data["Ingredients"] = np.add.reduce([data[x]+"," for x in ingredient_columns])
    data["Ingredients"] = data["Ingredients"]\
        .str.replace(",_", ",")\
        .str.replace(' {2,}', ' ')\
        .str.replace(',{2,}', ' ')\
        .str.strip()\
        .str.strip(",")
    
    data = data.drop(ingredient_columns, axis=1)
    
    vectorizer = CountVectorizer()
        
    # generate ingredient frequency matrix
    count_matrix = vectorizer.fit_transform(data["Ingredients"])
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                count_matrix.toarray(), 
                columns=vectorizer.get_feature_names()
            )
        ], axis=1)    
    
    return data, ingredient_type_map, other_columns

# Get Meal Plan Function
def get_meal_plan(
    ** args,
) -> List[str]:
    
    global current_meal_plan
    
    RANDOMIZATION_STRENGTH = 0.5
    
    recipes_per_meal_plan = args["recipes_per_meal_plan"]
    data = args["data"]
    saved_meal_plan = args["saved_meal_plan"]
    ingredient_type_map = args["ingredient_type_map"]
    other_columns = args["other_columns"]    
    
    # if passing a saved meal plan, skip the meal plan generation process
    if saved_meal_plan is not None and len(saved_meal_plan) > 0:
        new_meal_plan = data.query("`Recipe Name` in @saved_meal_plan").index.to_list()
    else:

        recipe_id_dict = data.reset_index().set_index("Recipe Name").loc[:,"index"].to_dict()
        num_columns = len(other_columns) + 2

        initial_recipes = [recipe_id_dict[key] for key, value in args.items() if key in recipe_id_dict and value]

        if len(initial_recipes) == 0:
            print("Select a Recipe or Ingredient to See a Meal Plan")
            return True

        ingredients = list(data.columns)[num_columns:]

        # build meal plan
        queue = [initial_recipes]
        meal_plans = []
        if len(initial_recipes) >= recipes_per_meal_plan:
            new_meal_plan = initial_recipes
        else:
            while len(queue) > 0:
                current_recipes = queue.pop()
                selected_recipes = data.iloc[current_recipes,num_columns:].sum(axis=0)
                selected_recipe_ingredients = list(selected_recipes[selected_recipes >= 1].index)
                sort_ascending = False
                filter_columns = selected_recipe_ingredients

                recipes_to_consider = data.query("index not in @current_recipes").iloc[:, 3:]

                intersection_weights = recipes_to_consider\
                    .loc[:,filter_columns]\
                    .sum(axis=1)

                union_weights = recipes_to_consider.sum(axis=1)

                random_weights = np.random.uniform(
                    low=1-RANDOMIZATION_STRENGTH, 
                    high=1+RANDOMIZATION_STRENGTH, 
                    size=(len(recipes_to_consider),)
                )

                potential_meal_plans = intersection_weights/union_weights * random_weights
                result = potential_meal_plans.sort_values(ascending=sort_ascending)\
                    .index.to_list()[0]

                new_meal_plan = current_recipes + [result]
                if len(new_meal_plan) == recipes_per_meal_plan:
                    new_meal_plan = sorted(new_meal_plan)
                else:
                    queue.append(new_meal_plan)

    # RESUME HERE IF SAVED MEAL PLAN
    selected_meal_plan_ingredients = data.query("index in @new_meal_plan")
    current_meal_plan = selected_meal_plan_ingredients["Recipe Name"].to_list()
    chart_data = selected_meal_plan_ingredients.melt(
        id_vars = other_columns + ["Recipe Name", "Ingredients"], 
        var_name = "Ingredient", 
        value_name="count"
    ).query("count > 0")
    
    chart_data["Ingredient_Type"] = chart_data["Ingredient"].replace(ingredient_type_map)    
    chart_data["Ingredient"] = chart_data["Ingredient"].str.replace("_"," ").str.title()
    
    ingredient_sort = chart_data.groupby(["Ingredient_Type", "Ingredient"])["count"].sum()\
    .reset_index().sort_values(["Ingredient_Type", "count"], ascending=False)["Ingredient"].to_list()
    chart = alt.Chart(chart_data).mark_rect().encode(
        x = alt.X(
            "Recipe Name", 
            axis = alt.Axis(orient="top", labelAngle=0),
            title = None
        ),
        y = alt.Y(
            "Ingredient", 
            axis = alt.Axis( labelAngle=0),
            sort = ingredient_sort,
            title = None
        ),
        color = alt.condition(alt.datum.count == 1, alt.Color("Ingredient_Type:N"), alt.value(None)),
        tooltip = ["Recipe Name"] + other_columns
    ).properties(
        width = 150 * len(selected_meal_plan_ingredients)
    )
    display(chart)
    
# Function to build UI elements    
def build_widgets(data, options_dict):
    
    global current_meal_plan
    
    # Build Search Widget
    default_search_text = "<search by recipe name and ingredients>"
    search_widget = wid.Text(placeholder = default_search_text)
    output_widget = wid.Output()
    options = [x for x in options_dict.values()]
    options_layout = wid.Layout(
        overflow='auto',
        border='1px solid black',
        width='300px',
        height='300px',
        flex_flow='column',
        display='flex'
    )

    @output_widget.capture()
    def on_checkbox_change(change):
        
        selected_recipe = change["owner"].description
        options_widget.children = sorted([x for x in options_widget.children], key = lambda x: x.value, reverse = True)
        
    for checkbox in options:
        checkbox.observe(on_checkbox_change, names="value")

    # Wire the search field to the checkboxes
    @output_widget.capture()
    def on_text_change(change):
        search_input = str.lower(change['new'].strip('').replace(' ', '_'))
        if search_input == '':
            # Reset search field
            new_options = sorted(options, key = lambda x: x.value, reverse = True)
        else:
            # Get matches by name
            # close_matches = [x for x in list(options_dict.keys()) if str.lower(search_input.strip('')) in str.lower(x)]
            close_matches = data.query("`Recipe Name`.str.contains(@search_input) or Ingredients.str.contains(@search_input)")["Recipe Name"].to_list()
            new_options = sorted(
                [x for x in options if x.description in close_matches], 
                key = lambda x: x.value, reverse = True
            )
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    
    # Build Save Widget
    save_button = wid.Button(
        description = "Save Plan"
    )
    
    saved_plans_dropdown = wid.Dropdown(
        options = [("<Select to generate new meal plan>", [])],
        description = "Saved Meal Plans",
        style={"description_width":"120px"},
        layout = wid.Layout(width="750px")
    )
    
    @output_widget.capture()
    def save_meal_plan(change):
        global current_meal_plan
        current_meal_plan_str = ", ".join(current_meal_plan)
        new_option = (current_meal_plan_str, current_meal_plan)
        saved_plans_dropdown.options = list(saved_plans_dropdown.options) + [new_option]
        saved_plans_dropdown.value = new_option[1]
    
    save_button.on_click(save_meal_plan)
    
    # Build Num Recipes Widget
    num_recipes_selector = wid.ToggleButtons(
        options=[3, 4, 5, 6, 7],
        value = 5,
        description='Number of Recipes:',
        style={"description_width":"120px"},
        layout = wid.Layout(width="100px")
    )
    
    # Define behavior for clearing save widget when other widgets changed
    @output_widget.capture()
    def clear_selected_meal_plan(change):
        #print(saved_plans_dropdown.options[0])
        saved_plans_dropdown.value = saved_plans_dropdown.options[0][1]
        
    num_recipes_selector.observe(clear_selected_meal_plan, names="value")
    for checkbox in options:
        checkbox.observe(clear_selected_meal_plan, names="value")
    
    # Compose UI
    options_widget = wid.VBox(options, layout=options_layout)
    multi_select = wid.VBox(
        [
            search_widget, 
            options_widget
        ]
    )
    
    display(output_widget)
    return multi_select, num_recipes_selector, saved_plans_dropdown, save_button

# Actually run the app, starting with a file upload widget
def run_app():
    file_output = wid.Output()
    file_widget = wid.FileUpload(accept=".csv")

    @file_output.capture()
    def get_uploaded_file(change):
        #print(change["new"])
        file_to_string = StringIO(
            str(
                change["new"][
                    list(file_widget.value.keys())[0]
                ]["content"],
                encoding = 'utf-8'
            )
        ) 

        raw_df = pd.read_csv(file_to_string)
        data, ingredient_type_map, other_columns = parse_data(raw_df)
        arg_dict = {
            title: wid.Checkbox(
                description=title, 
                value=False,
                style={"description_width":"0px"}
            ) for title in data["Recipe Name"].to_list()
        }

    #     ui = multi_checkbox_widget(data, arg_dict)
    #     num_recipes_selector = wid.ToggleButtons(
    #         options=[3, 4, 5, 6, 7],
    #         value = 5,
    #         description='Number of Recipes:',
    #         style={"description_width":"120px"},
    #         layout = wid.Layout(width="100px")
    #     )

        #save_button, saved_plans_dropdown = build_save_widgets()
        multi_select, num_recipes_selector, saved_plans_dropdown, save_button = build_widgets(data, arg_dict)

        # pass output of parse_data in as **args
        arg_dict["saved_meal_plan"] = saved_plans_dropdown
        arg_dict["recipes_per_meal_plan"] = num_recipes_selector
        arg_dict["data"] = wid.fixed(data)
        arg_dict["ingredient_type_map"] = wid.fixed(ingredient_type_map)
        arg_dict["other_columns"] = wid.fixed(other_columns)

        # compose UI
        search_ui = wid.VBox(
            [
                multi_select, 
                num_recipes_selector
            ]
        )    

        save_ui = wid.VBox(
            [
                wid.HBox(
                    [
                        saved_plans_dropdown, 
                        save_button
                    ]
                )
            ]

        )    

        out = wid.interactive_output(get_meal_plan, arg_dict)
        display(
            wid.HBox(
                [
                    search_ui, 
                    wid.VBox(
                        [
                            save_ui, 
                            out
                        ]
                    )
                ]
            )
        )

    file_widget.observe(get_uploaded_file, names="value")
    display(file_widget)
    display(file_output)