from topfarm.recorders import TopFarmListRecorder
import matplotlib.pyplot as plt

recorder = TopFarmListRecorder().load(r'E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\optimization_recorder_Revolution.pkl')
plt.figure()
plt.plot(recorder['counter'], recorder['AEP']/recorder['AEP'][-1])
plt.xlabel('Iterations')
plt.ylabel('AEP/AEP_opt')
plt.title('Optimization Progress: Revolution Wind Site')
plt.show()
print('done')
