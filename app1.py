import pandas as pd
import pandasgui
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
from kivy.uix.dropdown import DropDown

import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

data = [["Milk","Onion","Nutmeg","Kidney Beans","Eggs","Yogurt"],
        ["Dill","Onion","Nutmeg","Kidney Beans","Eggs","Yogurt"],
        ["Milk","Apple","Kidney Beans","Eggs"],
        ["Milk","Unicorn","Corn","Kidney Beans","Yogurt"],
        ["Corn","Onion","Onion","Kidney Beans","Ice Cream","Eggs"]]
te = TransactionEncoder()
te_arry = te.fit(data).transform(data)
df = pd.DataFrame(te_arry, columns=te.columns_)
items=list(te.columns_)

ratio=0.5  

class MyGrid(GridLayout):
    def __init__(self,**kwargs):
        super(MyGrid,self).__init__(**kwargs)
        self.cols=1

        

        self.inside=GridLayout()
        self.inside.cols=2

        self.add_widget(Label(text="Welcome To The Grocery Store"))
        self.add_widget(Label(text="Choose Any Item: Onion, Milk, Eggs, Nutmeg, Kidney Beans, Apple, Dill, Ice Cream, Corn, Unicorn, Yogurt"))

        self.inside.add_widget(Label(text="Select An Item: "))
        self.item=TextInput(multiline=False)
        self.inside.add_widget(self.item)

        self.add_widget(self.inside)
        self.submit=Button(text="Submit", font_size=40)
        self.submit.bind(on_press=self.pressed)
        self.add_widget(self.submit)

        

    def pressed(self,instance):
        purchase_item=self.item.text
        fpgrowth_df = fpgrowth(df, min_support=ratio, use_colnames=True)
        fpgrowth_df['itemsets'] = fpgrowth_df['itemsets'].apply(str)
        fpgrowth_df = fpgrowth_df[fpgrowth_df['itemsets'].str.contains(purchase_item)]
        gui1=pandasgui.show(fpgrowth_df)
        apriori_df = apriori(df, min_support=ratio, use_colnames=True, low_memory=True)
        apriori_df['itemsets'] = apriori_df['itemsets'].apply(str)
        apriori_df = apriori_df[apriori_df['itemsets'].str.contains(purchase_item)]
        gui2=pandasgui.show(apriori_df)
        self.item.text=""

class MyApp(App):
    def build(self):
        return MyGrid()

if __name__=="__main__":
    MyApp().run()