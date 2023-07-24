! pip install numpy==1.23.5
import pickle
import pandas as pd
import shap
from shap.plots._force_matplotlib import draw_additive_plot
import gradio as gr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
#import gradio.outputs as gro
from scipy import stats
from gradio import Interface
from gradio.components import Markdown, Row, Column, Slider, Dropdown, CheckboxGroup, Button, Textbox, Dataframe
import category_encoders
from category_encoders import TargetEncoder

#Load DFs

#Set up DF for filtered beers
bb_df = pd.read_csv('beer_brewery_imputed_githubtest.csv')

#Set up DF for percentiles (currently only includes beers with 500 or more reviews)
bb_df_percentile = pd.read_csv('bb_df_testing.csv')

#Set up DF for Aslin Examples
aslin_example_df_full = pd.read_csv('App_Example_Aslin.csv')
aslin_example_df = aslin_example_df_full.drop(['Brewery', 'Number of Ratings Beer'], axis=1)
aslin_example_df['ABV'] = aslin_example_df['ABV']*100

#Open individual Brewery list
with open('unique_brewery_file.pickle', 'rb') as file:
    unique_breweries_list = pickle.load(file)

#Define Choices
region_choices = ['Far West','Great Lakes','Mideast','Non-Con','Northeast','OTHER','Plains','Rocky Mountain','Southeast','Southwest']

style_choices = ['Altbier', 'Australian Sparkling Ale', 'Barleywine - American', 'Barleywine - English', 'Barleywine - Other', 'Belgian Blonde', 'Belgian Dubbel', 'Belgian Enkel / Patersbier', 'Belgian Quadrupel', 
    'Belgian Strong Dark Ale', 'Belgian Strong Golden Ale', 'Belgian Tripel', 'Bitter - Best', 'Bitter - Extra Special / Strong (ESB)', 'Bitter - Session / Ordinary', 'Bière de Champagne / Bière Brut', 
    'Black & Tan', 'Blonde Ale', 'Bock - Doppelbock', 'Bock - Eisbock', 'Bock - Hell / Maibock / Lentebock', 'Bock - Single / Traditional', 'Bock - Weizenbock', 'Bock - Weizendoppelbock', 'Brett Beer', 
    'Brown Ale - American', 'Brown Ale - Belgian', 'Brown Ale - English', 'Brown Ale - Imperial / Double', 'Brown Ale - Other', 'California Common', 'Chilli / Chile Beer', 'Cider - Dry', 'Cider - Graff', 
    'Cider - Herbed / Spiced / Hopped', 'Cider - Ice / Applewine', 'Cider - Other Fruit', 'Cider - Perry / Poiré', 'Cider - Rosé', 'Cider - Sweet', 'Cider - Traditional / Apfelwein', 'Corn Beer / Chicha de Jora', 
    'Cream Ale', 'Dark Ale', 'Farmhouse Ale - Bière de Coupage', 'Farmhouse Ale - Bière de Garde', 'Farmhouse Ale - Bière de Mars', 'Farmhouse Ale - Grisette', 'Farmhouse Ale - Other', 'Farmhouse Ale - Sahti', 
    'Farmhouse Ale - Saison', 'Festbier', 'Flavored Malt Beverage', 'Freeze-Distilled Beer', 'Fruit Beer', 'Gluten-Free', 'Golden Ale - American', 'Golden Ale - English', 'Golden Ale - Other', 'Grape Ale - Italian', 
    'Grape Ale - Other', 'Grodziskie / Grätzer', 'Hard Ginger Beer', 'Hard Kombucha / Jun', 'Hard Seltzer', 'Historical Beer - Adambier', 'Historical Beer - Broyhan', 'Historical Beer - Burton Ale', 'Historical Beer - Dampfbier', 
    'Historical Beer - Gruit / Ancient Herbed Ale', 'Historical Beer - Kentucky Common', 'Historical Beer - Kottbusser', 'Historical Beer - Kuit / Kuyt / Koyt', 'Historical Beer - Lichtenhainer', 
    'Historical Beer - Mumme', 'Historical Beer - Other', 'Historical Beer - Steinbier', 'Honey Beer', 'IPA - American', 'IPA - Belgian', 'IPA - Black / Cascadian Dark Ale', 'IPA - Brett', 
    'IPA - Brown', 'IPA - Brut', 'IPA - Cold', 'IPA - English', 'IPA - Farmhouse', 'IPA - Fruited', 'IPA - Imperial / Double', 'IPA - Imperial / Double Black', 'IPA - Imperial / Double Milkshake', 
    'IPA - Imperial / Double New England / Hazy', 'IPA - Milkshake', 'IPA - New England / Hazy', 'IPA - New Zealand', 'IPA - Other', 'IPA - Quadruple', 'IPA - Red', 'IPA - Rye', 'IPA - Session', 
    'IPA - Sour', 'IPA - Triple', 'IPA - Triple New England / Hazy', 'IPA - White / Wheat', 'Kellerbier / Zwickelbier', 'Koji / Ginjo Beer', 'Kvass', 'Kölsch', 'Lager - Amber / Red', 'Lager - American', 
    'Lager - American Amber / Red', 'Lager - American Light', 'Lager - Dark', 'Lager - Dortmunder / Export', 'Lager - Helles', 'Lager - IPL (India Pale Lager)', 'Lager - Japanese Rice', 'Lager - Leichtbier', 
    'Lager - Mexican', 'Lager - Munich Dunkel', 'Lager - Other', 'Lager - Pale', 'Lager - Strong', 'Lager - Vienna', 'Lager - Winter', 'Lambic - Framboise', 'Lambic - Fruit', 'Lambic - Gueuze', 'Lambic - Kriek', 
    'Lambic - Other', 'Lambic - Traditional', 'Malt Beer', 'Malt Liquor', 'Mead - Acerglyn / Maple Wine', 'Mead - Bochet', 'Mead - Braggot', 'Mead - Cyser', 'Mead - Melomel', 'Mead - Metheglin', 'Mead - Other', 
    'Mead - Pyment', 'Mead - Session / Short', 'Mead - Traditional', 'Mild - Dark', 'Mild - Light', 'Mild - Other', 'Märzen', 'Non-Alcoholic Beer - IPA', 'Non-Alcoholic Beer - Lager', 'Non-Alcoholic Beer - Other', 
    'Non-Alcoholic Beer - Porter / Stout', 'Non-Alcoholic Beer - Sour', 'Non-Alcoholic Beer - Wheat Beer', 'Old Ale', 'Other', 'Pale Ale - American', 'Pale Ale - Australian', 'Pale Ale - Belgian', 'Pale Ale - English', 
    'Pale Ale - Milkshake', 'Pale Ale - New England / Hazy', 'Pale Ale - New Zealand', 'Pale Ale - Other', 'Pale Ale - XPA (Extra Pale)', 'Pilsner - Czech / Bohemian', 'Pilsner - German', 'Pilsner - Imperial / Double', 
    'Pilsner - Italian', 'Pilsner - New Zealand', 'Pilsner - Other', 'Porter - American', 'Porter - Baltic', 'Porter - Coffee', 'Porter - English', 'Porter - Imperial / Double', 'Porter - Imperial / Double Baltic', 
    'Porter - Imperial / Double Coffee', 'Porter - Other', 'Pumpkin / Yam Beer', 'Rauchbier', 'Red Ale - American Amber / Red', 'Red Ale - Imperial / Double', 'Red Ale - Irish', 'Red Ale - Other', 'Roggenbier', 'Root Beer', 
    'Rye Beer', 'Rye Wine', 'Schwarzbier', 'Scotch Ale / Wee Heavy', 'Scottish Ale', 'Scottish Export Ale', 'Shandy / Radler', 'Smoked Beer', 'Sorghum / Millet Beer', 'Sour - Berliner Weisse', 'Sour - Flanders Oud Bruin', 
    'Sour - Flanders Red Ale', 'Sour - Fruited', 'Sour - Fruited Berliner Weisse', 'Sour - Fruited Gose', 'Sour - Other', 'Sour - Other Gose', 'Sour - Smoothie / Pastry', 'Sour - Traditional Gose', 'Specialty Grain', 
    'Spiced / Herbed Beer', 'Stout - American', 'Stout - Belgian', 'Stout - Coffee', 'Stout - English', 'Stout - Foreign / Export', 'Stout - Imperial / Double', 'Stout - Imperial / Double Coffee', 'Stout - Imperial / Double Milk', 
    'Stout - Imperial / Double Oatmeal', 'Stout - Imperial / Double Pastry', 'Stout - Imperial / Double White / Golden', 'Stout - Irish Dry', 'Stout - Milk / Sweet', 'Stout - Oatmeal', 'Stout - Other', 'Stout - Oyster', 
    'Stout - Pastry', 'Stout - Russian Imperial', 'Stout - White / Golden', 'Strong Ale - American', 'Strong Ale - English', 'Strong Ale - Other', 'Table Beer', 'Traditional Ale', 'Wheat Beer - American Pale Wheat', 
    'Wheat Beer - Dunkelweizen', 'Wheat Beer - Hefeweizen', 'Wheat Beer - Hefeweizen Light / Leicht', 'Wheat Beer - Kristallweizen', 'Wheat Beer - Other', 'Wheat Beer - Wheat Wine', 'Wheat Beer - Witbier / Blanche', 
    'Wild Ale - American', 'Wild Ale - Other', 'Winter Ale', 'Winter Warmer']

brewery_style_choices = ['Brew Pub', 'Cidery', 'Contract Brewery','Macro Brewery','Micro Brewery', 
                        'Nano Brewery', 'OTHER', 'Regional Brewery']

# brewery_style_choices = ['Bar / Restaurant / Store', 'Brew Pub', 'Cidery', 'Contract Brewery', 'Home / Non-Commercial Brewery', 'Macro Brewery', 'Meadery',
#     'Micro Brewery', 'Nano Brewery', 'OTHER', 'Regional Brewery']

state_choices = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MISSING', 'MN', 'MO', 'MS', 'MT', 'NC', 
    'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SAINT CROIX ISLAND VIRGIN ISLANDS', 'SAINT JOHN ISLAND VIRGIN ISLANDS', 'SAINT THOMAS ISLAND VIRGIN ISLANDS', 
    'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# Define flavors and hops
flavors_list = ['Apple', 'Apricot', 'Berry', 'Bitter', 'Caramel', 'Chocolate', 'Citrus', 'Clove', 'Coffee', 
                'Dry', 'Earthy', 'Fig', 'Floral', 'Fruity', 'Funky', 'Grapefruit', 'Hazelnut', 'Herbal', 
                'Malt', 'Nutmeg', 'Nutty', 'Peach', 'Pear', 'Peat', 'Pepper', 'Pine', 'Plum', 'Resin', 
                'Salty', 'Smoky', 'Sour', 'Spicy', 'Strawberry', 'Sweet', 'Tart', 'Toast', 'Toffee', 
                'Tropical', 'Vanilla']

hops_list = ['Amarillo', 'Cascade', 'Centennial', 'Chinook', 'Citra', 'Columbus', 'Crystal', 'Fuggle', 
             'Galaxy', 'Golding', 'Hallertau', 'Magnum', 'Mosaic', 'Noble', 'Nugget', 'Saaz', 'Simcoe', 'Tettnang', 
             'Warrior', 'Willamette']



# load the regressor model from disk
loaded_model_regressor = pickle.load(open("XGB_Untappd_regressor_FlavorBreakout.pkl", 'rb'))
# load the classification model from disk
loaded_model = pickle.load(open("XGB_Untappd_4_classifier_FlavorBreakout.pkl", 'rb'))



#Categorical Variable Encoder
loaded_enc_regressor = pickle.load(open("target_encoder_regressor_flavorbreakout.pkl", 'rb'))
loaded_enc_classification = pickle.load(open("target_encoder_classification_flavorbreakout.pkl", 'rb'))



# Setup SHAP
explainer = shap.Explainer(loaded_model_regressor) # PLEASE DO NOT CHANGE THIS.

#Set up similar beers function
def filter_beers(style, state):
    target_abv = 0.03  # Set the target ABV value within the function

    filtered_df = bb_df[(bb_df['Style'] == style) & (bb_df['State'] == state)].copy()
    filtered_df['ABV_diff'] = abs(filtered_df['ABV'] - target_abv)
    filtered_df.loc[filtered_df['ABV_diff'] > target_abv, 'ABV_diff'] = target_abv
    filtered_df = filtered_df[filtered_df['ABV_diff'] <= target_abv]
    sorted_df = filtered_df.sort_values(by='Number of Ratings Beer', ascending=False)
    limited_df = sorted_df.head(5)[['Brewery', 'Beer Name', 'Average Rating Beer', 'Number of Ratings Beer', 'Style', 'ABV', 'IBU', 'State']]
    
    limited_df = limited_df.rename(columns={'Average Rating Beer': 'Avg Rating', 'Number of Ratings Beer': '# Ratings'})
    
    limited_df['ABV'] = (limited_df['ABV'] * 100).round(2).astype(str) + '%'  
    limited_df['Avg Rating'] = limited_df['Avg Rating'].round(2)  
    limited_df['IBU'] = limited_df['IBU'].astype(int)
    limited_df['# Ratings'] = limited_df['# Ratings'].apply(lambda x: '{:,}'.format(x))  # Add commas to # Ratings
    
    return limited_df

# #Define percentiles before main function runs
# percentile_overall = 0
# percentile_state = 0
# percentile_style_overall = 0
# percentile_style_state = 0

def main_func(BeerName, ABV, IBU, Style, BreweryStyle, Region, State, Flavor_Group, Hop_Group):

    flavors_selected = [flavor for flavor in flavors_list if flavor in Flavor_Group]
    hops_selected = [hop for hop in hops_list if hop in Hop_Group]

    new_row = pd.DataFrame(columns=['ABV', 'IBU', 'Style', 'Brewery Style', 'Region', 'State'] + flavors_list + hops_list)
    new_row.loc[0] = [float(ABV), float(IBU), Style, BreweryStyle, Region, State] + [1 if flavor in Flavor_Group else 0 for flavor in flavors_list] + [1 if hop in Hop_Group else 0 for hop in hops_list]
    new_row[['ABV', 'IBU']] = new_row[['ABV', 'IBU']].astype(float)
    new_row['ABV'] = new_row['ABV']/100
    
    # Transform the new_row using the loaded encoder
    new_row_class = new_row
    new_row_regress = new_row
    new_row_encoded_class = loaded_enc_classification.transform(new_row_class)
    new_row_encoded_regressor = loaded_enc_regressor.transform(new_row_regress)
    
    prob = loaded_model.predict_proba(new_row_encoded_class)
    
    score_predict = loaded_model_regressor.predict(new_row_encoded_regressor)
    score_predict = score_predict[0]
    score_predict = round(score_predict, 2)
    score_predict_str = str(score_predict)
    score_predict_float = float(score_predict)
    
    
    #Build SHAP
    shap_values = explainer(new_row_encoded_regressor)
    
    plot = shap.plots.bar(shap_values[0], max_display=7, order=shap.Explanation.abs, show_data='auto', show=False)

    plt.tight_layout()
    local_plot = plt.gcf()
    plt.close()
    
    #Build Similar Beers
    similar_beers = filter_beers(Style, State) # fetch beers matching the style and state
    
    #Read in variables for percentiles
    nr_state_p = new_row['State'][0]
    nr_style_p = new_row['Style'][0]
    
    #Show Percentiles for prediction
    overall_df = bb_df_percentile
    state_df = bb_df_percentile[bb_df_percentile['State'] == nr_state_p]
    style_overall_df = bb_df_percentile[bb_df_percentile['Style'] == nr_style_p]
    style_state_df = bb_df_percentile[(bb_df_percentile['Style'] == nr_style_p) & (bb_df_percentile['State'] == nr_state_p)]
    
    # Calculate the percentile of a beer 

    percentile_overall = stats.percentileofscore(overall_df['Average Rating Beer'], score_predict_float).round(1)/100
    percentile_state = stats.percentileofscore(state_df['Average Rating Beer'], score_predict_float).round(1)/100
    percentile_style_overall = stats.percentileofscore(style_overall_df['Average Rating Beer'], score_predict_float).round(1)/100
    percentile_style_state = stats.percentileofscore(style_state_df['Average Rating Beer'], score_predict_float).round(1)/100
    
    percentile_dict0 = {
        "USA Overall": [percentile_overall],
        f"{nr_state_p} Overall": [percentile_state],
        f"USA by Style {nr_style_p}": [percentile_style_overall],
        f"{nr_state_p} by Style {nr_style_p}": [percentile_style_state]
    }    
    
    percentile_dict1 = {"Percentile Overall": percentile_overall} 
    percentile_dict2 = {f"Percentile {nr_state_p}": percentile_state} 
    percentile_dict3 = {f"Percentile {nr_style_p} Overall": percentile_style_overall} 
    percentile_dict4 = {f"Percentile {nr_style_p} {nr_state_p}": percentile_style_state} 
    
    #Convert to dataframe
    percentile_df = pd.DataFrame(
        { "type": ["USA Overall",f"{nr_state_p} Overall",f"USA by Style {nr_style_p}",f"{nr_state_p} by Style {nr_style_p}"],
          "value": [percentile_overall,percentile_state,percentile_style_overall,percentile_style_state],})
    
    

    return local_plot, similar_beers,score_predict_str,percentile_dict0



#"Below 4.0": float(prob[0][0]), "Above 4.0": 1-float(prob[0][0])},

#,percentile_overall, percentile_state, percentile_style_overall, percentile_style_state

##main_func('',.045, 41, 'IPA - Session', 'Micro Brewery', 'Far West', 'CA', [], [])    
    
## Create the UI
title = "<center><b>🍻 **Untappd Beer Rating Predictor**🍻</b></center>"
description1 = """
This app predicts beers scores based on Untappd data pulled in June 2023. Mean Average Error (MAE) of <b>.12</b> and Root Mean Squared Error (RMSE) of <b>.16</b>. 
The input variables in this model explain <b>65%</b> of the variation in the Untappd beer score """

# description2 = """
# To use the app, click on one of the examples, or adjust the values of the seven beer score predictors, and click on Analyze. ✨ 
# """ 

theme = gr.themes.Default()#primary_hue="amber"

with gr.Blocks(title=title, theme = theme) as demo:
    Markdown(f"# {title}")
    Markdown(description1)
    # Markdown("""---""")
    # Markdown(description2)
    # Markdown("""---""")
   
    submit_btn1 = gr.Button("Predict")
    with Row():        
        with Column():
            # BeerName = gr.components.Textbox(label='Beer Name (not required)', value = 'New Beer 1')
            # ABV = gr.components.Slider(label="ABV %", minimum=0, maximum=20, value=4.5, step=.1)
            # IBU = gr.components.Slider(label="IBU", minimum=0.0, maximum=200, value=41, step=1)
            # Style = gr.components.Dropdown(choices=style_choices, label="Select Beer Style", value= 'IPA - Session')
            # BreweryStyle = gr.components.Dropdown(choices=brewery_style_choices, label="Select Brewery Style", value= 'Micro Brewery')
            # Region = gr.components.Dropdown(choices=region_choices, label="Select USA Region", value= 'Far West')
            # State = gr.components.Dropdown(choices=state_choices, label="Select State", value= 'CA')
            # # Grouped checkboxes
            # Flavor_Group = gr.components.CheckboxGroup(choices=flavors_list, label="Select Flavors")
            # Hop_Group = gr.components.CheckboxGroup(choices=hops_list, label="Select Hops")
            
            
            BeerName = Textbox(label='Beer Name (not required)', value = 'New Beer 1')
            ABV = Slider(label="ABV %", minimum=0, maximum=20, value=4.5, step=.1)
            IBU = Slider(label="IBU", minimum=0.0, maximum=200, value=41, step=1)
            Style = Dropdown(choices=style_choices, label="Select Beer Style", value='IPA - Session')
            BreweryStyle = Dropdown(choices=brewery_style_choices, label="Select Brewery Style", value='Micro Brewery')
            Region = Dropdown(choices=region_choices, label="Select USA Region", value='Far West')
            State = Dropdown(choices=state_choices, label="Select State", value='CA')
            # Grouped checkboxes
            Flavor_Group = CheckboxGroup(choices=flavors_list, label="Select Flavors")
            Hop_Group = CheckboxGroup(choices=hops_list, label="Select Hops")
            
            
        #CREATE OUTPUTS
        with gr.Column(visible=True) as output_col:
            gr.Markdown("<h2><center><b>Untappd Score Prediction</b></center></h2>")
            score_predict_str = gr.Label(label="XGBoost Regressor")
        
            gr.Markdown("<h2><center><b>Prediction Drivers</b></center></h2>")
            local_plot = gr.Plot(label = 'Shap:')
            
            gr.Markdown("<h2><center><b>Percentiles for Beer</b></center></h2>")
            percentile_dict0 = gr.Label(label ='test', show_label=False)
            
            #percentile_df = gr.BarPlot(title = 'test bar plot', x ="type", y="value", vertical=False).style(container=False,)
            

            #label = gr.Label(label = "Predicted Label")
            
   
    submit_btn2 = gr.Button("Predict")
    
    Markdown("""---""")



    # Create a separate row for the output of filter_beers function
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h1><center><b>Similar Beers</b></center></h1>")
            #similar_beers_df = gro.Dataframe(label="", type="pandas")
            similar_beers = Dataframe(label="", type="pandas")
            

    #Button Click Events
    submit_btn1.click(
        main_func,
        [BeerName, ABV, IBU, Style, BreweryStyle, Region, State, Flavor_Group, Hop_Group],
        [local_plot,similar_beers,score_predict_str,percentile_dict0],
        api_name="Untappd_Rating_Model")
    
    submit_btn2.click(
        main_func,
        [BeerName, ABV, IBU, Style, BreweryStyle, Region, State, Flavor_Group, Hop_Group],
        [local_plot,similar_beers,score_predict_str,percentile_dict0],
        api_name="Untappd_Rating_Model1")

    #EXAMPLES   
    example_list = aslin_example_df.values.tolist()
    gr.Markdown("<h1><center><b>Aslin Beers Example Inputs</b></center></h1>")
    gr.Examples(example_list[:30],
                [BeerName, ABV, IBU, Style, BreweryStyle, Region, State, Flavor_Group, Hop_Group], # Flavor_Group, Hop_Group
                [local_plot, similar_beers, score_predict_str,percentile_dict0], 
                main_func, 
                cache_examples=True, label = "Aslin Beer List")
    


demo.launch()
