from deepchecks.tabular import datasets
from deepchecks.tabular.suites import data_integrity
from deepchecks.tabular import Dataset

# data = datasets.regression.avocado.load_data(data_format='DataFrame', as_train_test=False)
import pandas as pd

# def add_dirty_data(df):
#     # change strings
#     df.loc[df[df['type'] == 'organic'].sample(frac=0.18).index,'type'] = 'Organic'
#     df.loc[df[df['type'] == 'organic'].sample(frac=0.01).index,'type'] = 'ORGANIC'
#     # add duplicates
#     df = pd.concat([df, df.sample(frac=0.156)], axis=0, ignore_index=True)
#     # add column with single value
#     df['Is Ripe'] = True
#     return df


# dirty_df = add_dirty_data(data)
data = pd.read_csv(r"C:\karyalay\ModelDiagnostics\current2.csv")



# Categorical features can be heuristically inferred, however we
# recommend to state them explicitly to avoid misclassification.



ds = Dataset(data)







# Run Suite:
integ_suite = data_integrity()
suite_result = integ_suite.run(ds)
# Note: the result can be saved as html using suite_result.save_as_html()
# or exported to json using suite_result.to_json()
suite_result.save_as_html("testnaya2.html")