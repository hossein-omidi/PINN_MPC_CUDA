# PINN Based - Model Predictive Control of Single Zone Building
In this study, we developed Physics-Informed Neural Network (PINN) model for a single zone building. The developed model is used as the prediction model in the Model Predictive Controller (MPC). 

For MPC, Neural Generalized Predictive Control (NGPC), which uses Newton-Raphson to optimize the parameters within control horizon, is built from scratch by referring [1]

The code starts with generating data from single zone building model with baseline on/off control:

![basedline](https://github.com/PochingHsu/PINN-MPC/assets/165426535/88845c27-2da0-4cdd-bd3c-92674d718f26)

The data is divided into 70% for training amd 30% for testing.

The ANN prediction in testing (MSE of train: 0.0077980384, MSE of test: 0.0079830755, training time: 23.149399518966675):

![NN_test](https://github.com/PochingHsu/PINN-MPC/assets/165426535/9117b54e-4dd2-4667-9427-0587c6f4188a)

The PINN prediction in testing (MSE of train: 0.0082219969, MSE of test: 0.0027068360, training time: 48.96461486816406):

![PINN_test](https://github.com/PochingHsu/PINN-MPC/assets/165426535/9d4e5087-3a27-45fa-b994-22b8e1a5b1ce)

The PINN-MPC results (weight number: 0.3, Energy Consumption : 1.3954234467340754 [kWh] compared to baseline : 1.5066666666666668[kWh]):

![MPC](https://github.com/PochingHsu/PINN-MPC/assets/165426535/4a02eb91-9bda-4308-bee2-0f3c0302f2fa)

[1]	D. Soloway and P. J. Haley, “Neural generalized predictive control,” in Proceedings of the 1996 IEEE International Symposium on Intelligent Control, Dearborn, MI, USA: IEEE, 1996, pp. 277–282. doi: 10.1109/ISIC.1996.556214.
