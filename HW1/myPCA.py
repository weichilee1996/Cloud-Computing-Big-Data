#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def myPCA(dataset_path,target_dim):
    c_count_list = []
    temp = []
    c_list = []
    c_index = []  #可知第幾個是第幾類id
    data_matrix=[]
    e_vector = []
    output = []
    c_data_matrix = []
    target = ['views', 'likes', 'dislikes','comment_count', 'comments_disabled', 'ratings_disabled','video_error_or_removed']
    df = pd.read_csv(dataset_path)  
    #id range 1~43 but 其中只有16種
    for i in df['category_id']:
        c_index.append(i)
        if i not in c_list:
            c_list.append(i) 


    for i in df['category_id']:
        temp.append(i)
    for c in c_list:
        c_count_list.append(temp.count(c))
    c_count = list(zip(c_list,c_count_list)) #(種類,總數)


    #過濾features
    for feature in target:
        tmp = []
        for i in df[feature]:
            tmp.append(i)
        data_matrix.append(tmp)
    data_matrix = np.array(data_matrix).T



    for i in c_list:
        temp = []
        for j in range(len(c_index)):
            if i == c_index[j]:
                temp.append(data_matrix[j])  
        c_data_matrix.append(temp)


    #開始PCA
    for i in range(0,len(c_data_matrix)):
        c_data_matrix[i] = np.array(c_data_matrix[i])
        c_id=c_data_matrix[i].T
        z = np.dot(c_id,c_id.T)
        cov_matrix = np.cov(z)
        e_value,e_vector=np.linalg.eig(z)
        e_vector = e_vector[0:2]  #function時把2改成input dim
        out = np.dot(c_data_matrix[i],e_vector.T)
        #print(out.shape)
        output.append(out)
    output = np.array(output)

    return output

