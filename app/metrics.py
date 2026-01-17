from sklearn.metrics import classification_report

'''Converts classification metrics report into a machine-readable format.
It returns a dictionary of metrics for further processing or analysis.
By setting "output_dict=True", you're transforming what is usually a text-based table into a nested dictionary,
 which is much easier to use for logging or visualization.'''

def generate_metrics(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)
