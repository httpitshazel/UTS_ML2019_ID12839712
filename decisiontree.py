import csv
import numpy as np  
import math
import matplotlib.pyplot as plt
varnamestr = lambda v,nms: [ vn for vn in nms if id(v)==id(nms[vn])][0]  
class CUtileTool(object):  

    def dump_list(self, src_list, src_list_namestr):  
		#Print list line by line
		#return nothing
        print('\n============',src_list_namestr,'================')  
        list_len = len(src_list)  
        list_shape = np.shape(src_list)  
        print('type(',src_list_namestr,'):',type(src_list))    
        print('np.shape(',src_list_namestr,'):',np.shape(src_list))  
        if 1 == len(list_shape):  
            print(src_list)  
        elif 2 == len(list_shape):  
            for i in range(list_len):  
                if 0 == i:  
                    print('[',src_list[i])  
                elif (list_len - 1) == i:  
                    print(src_list[i],']')  
                else:  
                    print(src_list[i])  
        else:  
            print(src_list)  
        print('======\n')  
        return  
   
    def dump_array(self, src_a, src_dict_namestr):  
		#print array line by line
		#return nothing
        print('\n===============',src_dict_namestr,'===================')  
        a_len = len(src_a)  
        a_shape = np.shape(src_a)  
        print('type(',src_dict_namestr,'):',type(src_a))   
        print('np.shape(',src_dict_namestr,'):',np.shape(src_a))  
        if 1 == len(a_shape):  
            print(src_a)  
        elif 2 == len(a_shape):  
            for i in range(a_len):  
                if 0 == i:  
                    print('[',src_a[i])  
                elif (a_len - 1) == i:  
                    print(src_a[i],']')  
                else:  
                    print(src_a[i])  
        else:  
            print(src_a)  
        print('======\n')  
        return  
  
  
    def print_dict(self, src_dict, level, src_dict_namestr=''):  
		#print dict line by line
        if isinstance(src_dict, dict):  
            tab_str = '\t'  
            for i in range(level):  
                tab_str += '\t'  
            if 0 == level:  
                print(src_dict_namestr,'= {')  
            for key, value in src_dict.items():  
                if isinstance(value, dict):  
                    has_dict = False  
                    for k,v in value.items():  
                        if isinstance(v, dict):  
                            has_dict = True  
                    if has_dict:  
                        print(tab_str,key,":{")  
                        self.print_dict(value, level + 1)  
                    else:  
                        print(tab_str,key,':',value)  
                else:  
                    print(tab_str,key,': ',value,)  
            print(tab_str,'}')  
  
  
    def dump_dict(self, src_dict, src_dict_namestr):  
		#print dict line by line
		#return nothing
        print('\n===============',src_dict_namestr,'===================')  
        dict_len = len(src_dict)  
        dict_shape = np.shape(src_dict)  
        dict_type = type(src_dict)  
        print('len(',src_dict_namestr,'):',dict_len)  
        print('type(',src_dict_namestr,'):', dict_type)    
        print('np.shape(',src_dict_namestr,'):', dict_shape)  
        print('len(dict_shape):', len(dict_shape))  
          
        self.print_dict(src_dict, 0, src_dict_namestr)  
        print('======\n')  
        return  
          
    def dump(self, src_thing, src_thing_namestr):
        name = type(src_thing).__name__
        if name == 'list':
            return self.dump_list(src_thing, src_thing_namestr)  
        elif name == 'dict':
            return self.dump_dict(src_thing, src_thing_namestr)
        elif name == 'ndarray':
            print('hello')
            return self.dump_array(src_thing, src_thing_namestr)  
        else:
            print(src_thing_namestr,':', src_thing)  
        return

class CDictHelper(object):
	#dict helper

    def add(self, src_dict, key, value):
        if type(src_dict).__name__  != 'dict':
            print('error:expect type is dict, actual %', type(src_dict).__name__)
            return False
        src_dict[key] = value
        return True
    def add_func(self, src_dict, key, value):

        func = np.frompyfunc(self.add, 3, 1)
        return func(src_dict, key, value)

    def max_val_pair(self, src_dict, keys):
        if type(src_dict).__name__ != 'dict':
            print('type error:expace dict, acturl',type(src_dict).__name__)
            return None
        ret_key, max_val = None, -9527
        for key in keys:
            if key in src_dict.keys():
                if src_dict[key] > max_val:
                    ret_key, max_val = key, src_dict[key]
        return ret_key

    def min_val_pair(self, src_dict, keys):
        if type(src_dict).__name__ != 'dict':
            print('type error:expace dict, acturl',type(src_dict).__name__)
            return None
        ret_key, min_val = None, 9527
        for key in keys:
            if key in src_dict.keys():
                if src_dict[key] < min_val:
                    ret_key, min_val = key, src_dict[key]
        return ret_key

    def max_val_item(self, src_dict, parent_keys, child_key):
		#The judgment operator of the max function and the min function must have an equal decision, otherwise an incorrect decision tree will be obtained.
        if type(src_dict).__name__ != 'dict':
            print('type error:expace dict, acturl',type(src_dict).__name__)
            return None
        ret_key, max_val = None, -9527
        for key in parent_keys:
            if key in src_dict.keys() and child_key in src_dict[key]:

                if src_dict[key][child_key] > max_val:  
                    ret_key, max_val = key, src_dict[key][child_key]
        return ret_key

    def min_val_item(self, src_dict, parent_keys, child_key):
        if type(src_dict).__name__ != 'dict':
            print('type error:expace dict, acturl',type(src_dict).__name__)
            return None
        ret_key, min_val = None, 9527
        for key in parent_keys:
            if key in src_dict.keys() and child_key in src_dict[key]:

                if src_dict[key][child_key] <= min_val:
                        ret_key, min_val = key, src_dict[key][child_key]
        return ret_key
#Cutting specifications
#Assume that the optimal root node of the decision tree that has selected this sample set by the information gain is characterized by the gender
#1.Delete the gender feature value for all rows where Coed is located.
#2.Then delete the gender feature value column		
		
		
class CTailorSamples(object):
	#Crop sample
    def __init__(self, data_list, feat_type_list, feat_type_index, feat_value):
        self.data_list = data_list
        self.feat_type_list = feat_type_list
        self.feat_type_index_tailed = feat_type_index
        self.feat_value_tailed = feat_value
        self.tailer_work()

    def get_samples(self):
		#Return the cropped sample
        return self.data_list, self.feat_type_list

    def get_all_indexs(self, src_list, dst_value):

        dst_val_index = [i for i,x in enumerate(src_list) if x == dst_value]
        return dst_val_index
    def tailer_work(self):

        del self.feat_type_list[self.feat_type_index_tailed]
		#Extract the deleted feature column
        colum_to_del = self.feat_type_index_tailed
        self.feat_value_list = [sample[colum_to_del] for sample in self.data_list]
		#Find the subscript of the line containing all the samples of the self.feat_value_tailed
        rows_to_del = self.get_all_indexs(self.feat_value_list, self.feat_value_tailed)
		#Delete the row of elf.src_data_list corresponding to the row index in row_index_list
        rows_to_del.reverse()
        for row in rows_to_del:
            del self.data_list[row]
		# Delete the given feature column
        for row in range(len(self.data_list)):
            del self.data_list[row][colum_to_del]
        return self.data_list, self.feat_type_list

class CID3DecisionTree(object): 

    def __init__(self, data_list, feat_list, leastFeatNum):  
		#Initialization function
        #param data_list: data set
        #param feat_list: list of feature types for the data
        #return none
        self.data_list = data_list  
          

        self.feat_list = feat_list  
        

        self.leastFeatNum = leastFeatNum
        

        self.samples_shanon_entropy = 0.0
          

        self.n_feats = len(feat_list)
 
        self.feat_value_list = []

        self.class_list = []  
        
        self.stat_dict = {}


        self.tree_dict = {}

        self.pickout_feat()
        self.pickout_class()
        self.pickout_samples_shannon_entropy()
        self.build_stat_dict()
        return  

    def get_sample_cnt(self, feat_values, val, class_values, label):
		#feat_values = ['N','N','N','Y']
		#class_values = ['N','N','Y','N']
		#When the same index, the feature value is high and the classification value is the total number of refuses is 2
        if type(feat_values).__name__ != 'list':
            print('type error:param1 expect list, actual', type(feat_values).__name__)
            return None
        if type(class_values).__name__ != 'list':
            print('type error:param2 expect list, actual', type(feat_values).__name__)
            return None
        if len(feat_values) != len(class_values):
            print('len error:param1 and param2 are of different length')
            return None
        
        ret_cnt = 0
        for i in range(len(feat_values)):
            pair_tuple = (feat_values[i], class_values[i])
            if pair_tuple == (val, label):
                ret_cnt += 1
        return ret_cnt
        
    def shan_ent_ele(self, p):
        if 0 == p:  
            return 0
        else:  
            return -1 * p * math.log2(p)
    def shan_ent(self, P):
		#Sample probability
		#Extract the eigenvalues of each type of feature
        func = np.frompyfunc(self.shan_ent_ele, 1, 1)  
        ent_ele_list = func(P)  
        entropy = ent_ele_list.sum() 
        return entropy
          
    def pickout_feat(self):  
		#Extract the classification list
        self.feat_value_list = []
        for dim in range(self.n_feats):  
            curtype_feat = [sample[dim] for sample in self.data_list]   
            self.feat_value_list.append(curtype_feat)
        return self.feat_value_list

    def pickout_class(self): 

        self.class_list = [sample[-1] for sample in self.data_list]  
        return self.class_list
      
    def pickout_samples_shannon_entropy(self):  
		#Calculate the entropy of the samples
		#Classification label distribution of statistical sample sets
        label_set = set(self.class_list) 
        label_cnt_list = [] 
        for label in label_set:  
            label_cnt_list.append(self.class_list.count(label))  
		#Statistical sample set classification label probability density
        n_samples = len(self.class_list)  
        label_prob_list = [label_cnt/n_samples for label_cnt in label_cnt_list]  
		#Calculating the entropy of the sample
        self.samples_shanon_entropy = self.shan_ent(label_prob_list)  
        return self.samples_shanon_entropy

    def get_stat_dict(self):
        return self.stat_dict
    def build_stat_dict(self):
		#Core function
        dh = CDictHelper()
        self.stat_dict= dict({})
        self.stat_dict['samples_ent'] = self.samples_shanon_entropy
        class_values, class_set = self.class_list, set(self.class_list)
        for i in range(self.n_feats):
            feat, cond_ent, info_gain = self.feat_list[i], 0, 0 
            feat_values, value_set = self.feat_value_list[i], set(self.feat_value_list[i])
            for val in value_set:
                cnt, p_self = feat_values.count(val), feat_values.count(val)*1.0/len(feat_values)
                labels, s_dist, p_name, p_dens= [], [], [], []
                for label in class_set:
                    labels.append(label)
                    n_label = self.get_sample_cnt(feat_values, val, class_values, label)
                    s_dist.append(n_label)
                    p_dens.append(n_label * 1.0 / cnt)
                    p_name.append('p_' + label)
                shan_ent = self.shan_ent(p_dens)
                cond_ent += p_self * shan_ent
                info_gain = self.samples_shanon_entropy - cond_ent

                if 'samples_ent' not in self.stat_dict.keys():
                    self.stat_dict['samples_ent'] = self.samples_shanon_entropy
                if feat not in self.stat_dict.keys():
                    self.stat_dict[feat] = {}
                dh.add(self.stat_dict[feat], val, dict({}))
                dh.add_func(self.stat_dict[feat][val], ['cnt', 'p_self', 'ent'], [cnt, p_self, shan_ent])
                for i in range(len(class_set)):
                    if labels[i] not in self.stat_dict[feat][val].keys():
                        self.stat_dict[feat][val][labels[i]] = {}
                    self.stat_dict[feat][val][labels[i]]['cnt']    = s_dist[i]
                    self.stat_dict[feat][val][labels[i]]['p_self'] = p_dens[i]
                pass
            dh.add_func(self.stat_dict[feat], ['cond_ent', 'info_gain'], [cond_ent, info_gain])
            pass
        pass
        return self.stat_dict

    def get_tree_dict(self):
        return self.tree_dict
    def create_tree(self):

        dh = CDictHelper()

        root = dh.max_val_item(self.stat_dict, self.feat_list, 'info_gain')
        feat, feat_ind = root, self.feat_list.index(root)

        val_set = set(self.feat_value_list[feat_ind])
        rcond = value = dh.min_val_item(self.stat_dict[feat], val_set, 'ent')
        lcond = dh.max_val_item(self.stat_dict[feat], val_set, 'ent')

        class_set = set(self.class_list)
        rnode = dh.max_val_item(self.stat_dict[feat][value], class_set, 'p_self')
        lnode = dh.min_val_item(self.stat_dict[feat][value], class_set, 'p_self')

        if self.n_feats >= self.leastFeatNum:
            tailor = CTailorSamples(self.data_list, self.feat_list, feat_ind, value)
            new_samples_list, new_feat_list = tailor.get_samples()
            child_samples = CID3DecisionTree(new_samples_list, new_feat_list, self.leastFeatNum)
            lnode = child_samples.create_tree()
            
		#Fill decision tree
        self.tree_dict = {}
        self.tree_dict[root] = {}       
        self.tree_dict[root][rcond] = rnode
        self.tree_dict[root][lcond] = lnode
        return self.tree_dict


decisionNode = dict(boxstyle="round4", color='r', fc='0.9')

leafNode = dict(boxstyle="circle", color='m')

arrow_args = dict(arrowstyle="<-", color='g')
def plot_node(node_txt, center_point, parent_point, node_style):

    createPlot.ax1.annotate(node_txt, 
                            xy=parent_point,
                            xycoords='axes fraction',
                            xytext=center_point,
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=node_style,
                            arrowprops=arrow_args)

def get_leafs_num(tree_dict):


    leafs_num = 0
    if len(tree_dict.keys()) == 0:
        print('input tree dict is void!!!!!')
        return 0

    root = list(tree_dict.keys())[0]

    child_tree_dict =tree_dict[root]
    for key in child_tree_dict.keys():

        if type(child_tree_dict[key]).__name__=='dict':

            leafs_num += get_leafs_num(child_tree_dict[key])
        else:

            leafs_num += 1

    return leafs_num

def get_tree_max_depth(tree_dict):


    max_depth = 0
    if len(tree_dict.keys()) == 0:
        print('input tree_dict is void!')
        return 0

    root = list(tree_dict.keys())[0]

    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():

        this_path_depth = 0

        if type(child_tree_dict[key]).__name__ == 'dict':

            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])
        else:

            this_path_depth = 1
        if this_path_depth > max_depth:
            max_depth = this_path_depth

    return max_depth

def plot_mid_text(center_point, parent_point, txt_str):

    x_mid = (parent_point[0] - center_point[0])/2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1])/2.0 + center_point[1]
    createPlot.ax1.text(x_mid, y_mid, txt_str)
    return

def plotTree(tree_dict, parent_point, node_txt):

    leafs_num = get_leafs_num(tree_dict)
    root = list(tree_dict.keys())[0]

    center_point = (plotTree.xOff+(1.0+float(leafs_num))/2.0/plotTree.totalW,plotTree.yOff)

    plot_mid_text(center_point, parent_point, node_txt)

    plot_node(root, center_point, parent_point, decisionNode)

    child_tree_dict = tree_dict[root]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD

    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            plotTree(child_tree_dict[key],center_point,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plot_node(child_tree_dict[key],
                      (plotTree.xOff,plotTree.yOff),
                      center_point,leafNode)
            plot_mid_text((plotTree.xOff,plotTree.yOff),center_point,str(key))

    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD
    return

def createPlot(tree_dict):
	#Drawing a decision tree
    fig=plt.figure(1,facecolor='white')

    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1=plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW=float(get_leafs_num(tree_dict))
    plotTree.totalD=float(get_tree_max_depth(tree_dict))
    if plotTree.totalW == 0:
        print('tree_dict is void~')
        return
    plotTree.xOff=-0.5/plotTree.totalW;
    plotTree.yOff=1.0;
    plotTree(tree_dict, (0.5,1.0), '')
    plt.show()

def create_samples():  
    from google.colab import files
    
    csv_file = files.upload()
    csv_reader_lines = csv.reader(csv_file)
    data_list = [[ '1_6',  'N',   'Coed', 'N', 'N'],  
                 [ '1_6',  'N',   'Coed', 'N', 'Y'],  
                 [ '1_6',  'N',   'Coed', 'Y', 'N'],  
                 [ '1_6',  'Y',   'Coed', 'N', 'Y'],  
                 [ '1_12',  'N',   'Coed', 'N','N'],
                 [ '1_12',  'N',   'Coed', 'Y','N'],  
                 [ '1_12',  'Y',   'Coed', 'Y','N'],  
                 [   '1_2', 'N',   'Coed', 'N','N'],  
                 [   '1_2',  'Y',  'Coed', 'N','N'],  
                 [   'M_H',  'N',  'Coed', 'N','Y'],  
                 [ 'EEC',    'N',  'Coed', 'N','N'],  
                 [ 'HS',   'N',  'Coed',  'N', 'N'],  
                 [ 'BD',   'N', 'Coed',   'N', 'N'],  
                 ['7_12', 'N', 'Boys',    'N', 'N'],  
                 ['7_12', 'N',  'Girls',  'N', 'N']]
    #school = 0
    #for one_line in csv_reader_lines:
        #try:
          #data_list.append(one_line) 
          #school = school + 1
        #except:
          #pass
    
    feat_list = [ 'type', 'Preschool_ind', 'gender','Late_opening']  
    return data_list, feat_list

if __name__=='__main__':  
	
    train_data_list, feat_list = create_samples()
	#Decision tree generates the conditions for termination. When the characteristics of the sample are less than leastFeatNum, the decision tree of the sample set is no longer sought.
    leastFeatNum = 2
	#Create ID3 algorithm to generate the object of the decision tree class
    samples = CID3DecisionTree(train_data_list, feat_list, leastFeatNum)
	#Decision tree dictionary
    samples.create_tree()
	#Print the statistical dictionary of the samples
    CUtileTool().dump(samples.get_stat_dict(), 'samples statistic dict')
	#Print decision tree dictionary
    CUtileTool().dump(samples.get_tree_dict(), 'samples decision-tree dict')
	#Drawing a decision tree
    createPlot(samples.get_tree_dict())