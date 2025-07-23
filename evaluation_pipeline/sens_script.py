import json
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle


# Defining the function
def get_plots(type, json_file, name):
    if type == "default":
        pattern = [1,1,1,0,0,0]
    elif type == "flipped":
        pattern = [0,0,0,1,1,1]
    else:
        return print(f"""Please provide correct lable for second argument. It could be "default" or "flipped""")
    
    # Loading the file
    # with open(json_file,'r') as file:
    #     Score = json.load(file)
    with open(json_file, "rb") as file:
        Score = pickle.load(file)


    all_score = []
    for i in Score: # Either one of below line will be activated !!!!!!!!!!!!!!!!!!!!
        # avg_score = (Score[i]['content_score'] + Score[i]['structure_score'])/2
        # avg_score = Score[f"{i}"]['cells'][str(name)]['f1']
        avg_score=i['table_score']
        # avg_score = Score[i]
        all_score.append(avg_score)

    # Final code here...
    des_array = np.tile(pattern, 50)
    # Example ground truth labels and scores:
    y_true = des_array  # 0: similar, 1: different
    y_scores =np.array(all_score)
    y_scores_min = np.min(y_scores)
    y_scores_max = np.max(y_scores)
    y_scores = (y_scores - y_scores_min)/(y_scores_max-y_scores_min)
   
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    # Optionally, compute the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    print("FPR:", fpr)
    print("TPR:", tpr)
    print("Thresholds:", thresholds)
    print("AUC:", roc_auc)
    youden_j = tpr - fpr
    # youden_j = np.float32([0.5])
    # Find the index of the maximum J statistic
    optimal_index = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_index]
    optimal_j = youden_j[optimal_index]

    print("Optimal Threshold:", optimal_threshold)
    print("Maximum Youden's J:", optimal_j)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.scatter(fpr[optimal_index], tpr[optimal_index], color='red', label='Optimal threshold = %0.2f' % optimal_threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


    if type == "default":
        predictions = np.where(y_scores >= optimal_threshold, 1, 0)
    elif type == "flipped":
        predictions = np.where(y_scores >= optimal_threshold, 0, 1)


    y_true = des_array
    y_pred = predictions

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}, F1: {f1}")


    # Extract TN, FP, FN, TP from the confusion matrix
    TN, FP, FN, TP = cm.ravel()

    # Compute sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"Sensitivity (TPR): {sensitivity:.2f}")
    print(f"Specificity (TNR): {specificity:.2f}")

    # Visualize the sensitivity and specificity using a bar chart
    metrics = ['Sensitivity', 'Specificity']
    values = [sensitivity, specificity]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen'])
    plt.ylim(0, 1)
    plt.ylabel('Rate')
    plt.title('Sensitivity and Specificity')

    # Annotate the bars with the metric values
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{value:.2f}', 
                ha='center', va='bottom', fontweight='bold')

    plt.show()
    return


