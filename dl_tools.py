# 数据向量化（非常重要）
import numpy as np
import matplotlib.pylab as plt

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(loss)+1)
    
    plt.figure()
    plt.plot(epochs,loss,'bo',label='Training loss',color='blue')
    plt.plot(epochs,val_loss,'bo',label='Validation loss',color='red')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.figure()
    plt.plot(acc,'bo',label='Training acc',color='blue')
    plt.plot(val_acc,'bo',label='Validation acc',color='red')
    plt.title('Traing and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
def k_metrics_train(dl_model,train_data,train_targets,k=4,num_epochs=100):
    num_val_samples = len(train_data)//k
    all_scores=[]
    
    for i in range(k):
        print('processing field NO.',i)
        val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
        partical_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
        train_data[(i+1)*num_val_samples:]],
        axis=0)
        partical_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
        train_targets[(i+1)*num_val_samples:]],
        axis=0)
        model=dl_model
        print(u"开始训练————")
        print('训练中————')
        cost = model.fit(partical_train_data,
                     partical_train_targets,
                     validation_data = (val_data,val_targets),
                     epochs=num_epochs,
                     batch_size=1,
                     verbose=0)
        print("本轮训练完毕")
        mae_history = cost.history['val_mean_absolute_error']
        all_scores.append(mae_history)
    