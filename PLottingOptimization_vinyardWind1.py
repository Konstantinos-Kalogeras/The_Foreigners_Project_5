from topfarm.recorders import TopFarmListRecorder
import matplotlib.pyplot as plt

######################################################################################################################################
############################# This project was developed with the help of Prof. Rafael Vallota Rodrigues and ChatGPT #################
######################################################################################################################################

######################################################################################################################################
############################# Developer: Konstantinos Kalogeras ###################################################################### 
############################# Project Partner: Dat Trinh #############################################################################
######################################################################################################################################

# running of Optimization and Recordoing plotted by Konstantinos Kalogeas

# Optimization code developed by Prof. Rodrigues, adjustment needed to made to the Proffessors code to adjusted to specific circumstances
# for each wind farm.  

# this optimization was not done using partials

recorder_2 = TopFarmListRecorder().load(r'E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\optimization_VinyardWind1_2_Derivative_version.pkl')


plt.figure()
plt.plot(recorder_2['counter'], recorder_2['AEP']/recorder_2['AEP'][-1])
plt.xlabel('Iterations')
plt.ylabel('AEP/AEP_opt')
plt.title('Optimization Progress: Vineyard Wind 1')
plt.show()