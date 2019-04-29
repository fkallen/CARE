import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# leakyReLU: who are you?
def prelu(_x): # I am you, but stronger
  with tf.variable_scope('prelu', reuse=tf.AUTO_REUSE):
    alpha = tf.get_variable('alpha', [1],
                       initializer=tf.constant_initializer(0.1),
                       dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alpha * (_x - abs(_x)) * 0.5

  return pos + neg

def make_bootstrap(X, Y):
    i = np.random.choice(Y.shape[0], Y.shape[0])
    return X[i], Y[i], np.delete(X, i, 0), np.delete(Y, i, 0)

def train(model_path, train_paths):
    features_total = np.concatenate([np.load(path+'_X.bin.npy') for path in train_paths], 0)
    labels_total = np.concatenate([np.load(path+'_Y.bin.npy') for path in train_paths], 0)
    print("features_total.shape:", features_total.shape)
    print("labels_total.shape:", labels_total.shape)
    num_total = labels_total

    ### data preparation

    # The last 5 million have MUCH fewer zero labels than the average
    # so that's why I'm splitting the set like this
    np.random.seed(42)
    test_indices = np.random.choice(num_total, features_total.shape[0]//10, replace=False) # select 10% pseudo-random indices
    train, test = np.delete(features_total, test_indices, 0), features_total[test_indices]
    labels_train, labels_test = np.expand_dims(np.delete(labels_total, test_indices, 0),-1), np.expand_dims(labels_total[test_indices],-1)
    np.random.seed()
    # Dataset is really large so I'm only taking one bootstrap
    bootstrap, labels_bootstrap, validate, labels_validate = make_bootstrap(train, labels_train)
    print("bootstrap.shape:", bootstrap.shape)
    print("labels_bootstrap.shape:", labels_bootstrap.shape)
    print("validate.shape:", validate.shape)
    print("labels_validate.shape:", labels_validate.shape)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(training_init_op, feed_dict={x:bootstrap, y:labels_bootstrap})

        batch_count, ckpt_count, print_interval = 0, 0, (bootstrap.shape[0]//batch_size)//4
        for it in tqdm(range(0,num_epochs*bootstrap.shape[0], batch_size)):
        #for it in range(0,num_epochs*bootstrap.shape[0], batch_size):
            sess.run(step, feed_dict={train_toggle:True})
            if batch_count % print_interval == 0:
                sess.run(validation_init_op, feed_dict={x:validate, y:labels_validate})
                validate_predictions = np.zeros([labels_validate.shape[0],1])
                for it_ in tqdm(range(0,validate.shape[0],test_size)):
                #for it_ in range(0,validate.shape[0],test_size):
                    batch_predictions = sess.run(prediction, feed_dict={train_toggle:False})
                    validate_predictions[it_:it_+batch_predictions.shape[0]] = batch_predictions
                fpr_, tpr_, _ = roc_curve(labels_validate[:,0], validate_predictions[:,0])

                plt.figure(1)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_, tpr_, label="MLP")

                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC curve')
                plt.legend(loc='best')
                plt.savefig(model_path[:-5]+"_ROC_"+str(ckpt_count).zfill(4)+".png")
                plt.clf()
                saver = tf.train.Saver()
                save_path = saver.save(sess, model_path[:-5]+"_"+str(ckpt_count).zfill(4)+".ckpt")
                sess.run(training_init_op, feed_dict={x:bootstrap, y:labels_bootstrap})
                ckpt_count+=1
            batch_count+=1

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("Model saved in path: %s" % save_path)

    # #rf = RandomForestClassifier(n_estimators=64, max_depth=12, min_samples_split=2, random_state=42, n_jobs=64)
    # #rf.fit(bootstrap, labels_bootstrap[:,0])
    # #saved_rf = pickle.dump(rf, open("trained_forest", "wb"))
    # #rf = pickle.load(open("trained_forest", "rb"))
    # #rf_predictions = rf.predict_proba(validate)

    # #fpr, tpr, _ = roc_curve(labels_validate[:,0], rf_predictions[:,-1])
    # fpr_, tpr_, _ = roc_curve(labels_validate[:,0], validate_predictions[:,0])

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # #plt.plot(fpr, tpr, label='RF')
    # plt.plot(fpr_, tpr_, label="MLP")

    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.savefig("ROC_03-21.png")
    # #plt.show()

    # # for asdf in range(100):
    # #     i_0 = np.random.choice(np.argwhere(labels_validate==0)[:,0], 5)
    # #     i_1 = np.random.choice(np.argwhere(labels_validate==1)[:,0], 5)
    # #     i = np.concatenate([i_0,i_1])

    #     #print(validate_predictions[i,0])
    #     #print(rf_predictions[i])

class Classifier(object):
    def __init__(self, model_path):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        print("Model restored.")
    
    def infer(self, inf_features):
        inf_predictions = np.zeros([len(inf_features)])
        self.sess.run(validation_init_op, feed_dict={x:inf_features, y:np.zeros([len(inf_features), 1])})
        #for it in tqdm(range(0,len(inf_features), test_size)):
        for it in range(0,len(inf_features), test_size):
            predictions_ = self.sess.run(prediction, feed_dict={train_toggle:False})
            inf_predictions[it:it+predictions_.shape[0]] = predictions_[:,0]
        return list(inf_predictions)

    def __del__(self):
        self.sess.close()

### network definition
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

num_epochs, batch_size, test_size = 30, 2**8, 2**15
dtype, in_shape, l_shape = tf.float32, [None, 136], [None, 1]
x = tf.placeholder(dtype=dtype, shape=in_shape)
y = tf.placeholder(dtype=dtype, shape=l_shape)
train_toggle = tf.placeholder(dtype=tf.bool, shape=[]) # batch-normalization training toggle


train_set = tf.data.Dataset.from_tensor_slices((x, y)).repeat(num_epochs).batch(batch_size)
val_set = tf.data.Dataset.from_tensor_slices((x, y)).batch(test_size)
ds_iter = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
training_init_op = ds_iter.make_initializer(train_set)
validation_init_op = ds_iter.make_initializer(val_set)
X, Y = ds_iter.get_next()

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

def conv_cell(input):
    result = tf.layers.conv2d(input, 64, [3,3], padding="SAME", activation=prelu)
    result = tf.layers.batch_normalization(result, training=train_toggle)
    result = tf.layers.conv2d(result, 64, [5,5], padding="SAME")
    result = result + input
    result = tf.layers.batch_normalization(result, training=train_toggle)
    return result

net = tf.reshape(X, [-1, 4, 17, 2])
print("INIT", net.get_shape().as_list())

net = tf.layers.conv2d(net, 64, [1,3], padding="SAME", activation=prelu)
print("INIT CONV", net.get_shape().as_list())

net = tf.layers.batch_normalization(net, training=train_toggle)
net = tf.layers.max_pooling2d(net, [1,3], [1,1], "SAME")
print("INIT POOL", net.get_shape().as_list())

net = conv_cell(net)
print("FIRST CELL", net.get_shape().as_list())

net = tf.layers.max_pooling2d(net, [1,3], [1,1], "SAME")
print("FIRST POOL", net.get_shape().as_list())

net = conv_cell(net)
print("SECOND CELL", net.get_shape().as_list())

net = tf.layers.max_pooling2d(net, [1,3], [1,1], "SAME")
print("SECOND POOL", net.get_shape().as_list())

net = conv_cell(net)
print("THIRD CELL", net.get_shape().as_list())

net = tf.layers.max_pooling2d(net, [1,3], [1,1], "SAME")
print("THIRD POOL", net.get_shape().as_list())

net = conv_cell(net)
print("FOURTH CELL", net.get_shape().as_list())

net = tf.layers.max_pooling2d(net, [1,3], [1,1], "SAME")
print("FOURTH POOL", net.get_shape().as_list())


net = tf.reshape(net, [-1, 4 * 17 * 64])
# output layer
net = tf.layers.dense(net, 1, activation=None)
prediction = tf.sigmoid(net)

loss = tf.losses.sigmoid_cross_entropy(Y, net)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    step = tf.train.AdamOptimizer().minimize(loss)







