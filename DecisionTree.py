def gini_index(groups,classes):
  #count all the samples at the spiliting point
  n_instances = float(sum([len(group) for group in groups]))
  gini = .0
  for group in groups:
    size = len(group)
    if size == 0 :
      continue
    score = .0
    for class_val in classes:
      p = [row[-1] for row in group].count(class_val) / size
      score += p * p
    gini += (1.0 - score) * (size/n_instances)
  return gini

#  # test Gini values
# print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
# print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

def test_split(index,th_value,dataset):#column_number,threshold_value,original_dataset
  left , right = list() , list()
  for row in dataset:
    if row[index] < th_value:
      left.append(row)
    else:
      right.append(row)
  return left, right