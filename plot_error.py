import matplotlib
import matplotlib.pyplot as plt

error_file = open('error.txt', 'r')
  
epoch = 0
error = []
epoch_list = []
  
while True: 
  
    line = error_file.readline() 

    if not line: 
        break
    
    if "Test set: Average loss:" in line:
        split = line.split()
        loss = split[4].replace(',', '')
        test_good_pred, test_len = line.split()[6].split('/')
        accuracy = int(test_good_pred)/int(test_len) * 100
        error.append(100 - accuracy)
        epoch_list.append(epoch)
        epoch = epoch + 1


#make plot
fig, ax = plt.subplots()
ax.plot(epoch_list, error)

ax.set(xlabel='epoch', ylabel='error (%)',
       title='')
ax.grid()

plt.show()

error_file.close() 