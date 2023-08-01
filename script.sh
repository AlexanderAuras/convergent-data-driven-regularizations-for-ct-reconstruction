conda activate FSDLIP


#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_X_gauss_elli model=fbp noise_level=0.01 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_X_uni_elli model=fbp noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_X_gauss_lodopa model=fbp noise_level=0.01 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_X_uni_lodopa model=fbp noise_level=0.01 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_X_gauss_elli model=tikhonov noise_level=0.01 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_X_uni_elli model=tikhonov noise_level=0.01 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_X_gauss_lodopa model=tikhonov noise_level=0.01 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
#/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_X_uni_lodopa model=tikhonov noise_level=0.01 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform


/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_none_gauss_elli model=fbp noise_level=0.0 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_low_gauss_elli model=fbp noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_med_gauss_elli model=fbp noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_high_gauss_elli model=fbp noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_low_uni_elli model=fbp noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_med_uni_elli model=fbp noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_high_uni_elli model=fbp noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform

/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_none_gauss_lodopa model=fbp noise_level=0.0 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_low_gauss_lodopa model=fbp noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_med_gauss_lodopa model=fbp noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_high_gauss_lodopa model=fbp noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_low_uni_lodopa model=fbp noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_med_uni_lodopa model=fbp noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fbp_high_uni_lodopa model=fbp noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform


/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_none_gauss_elli model=tikhonov noise_level=0.0 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_gauss_elli model=tikhonov noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_gauss_elli model=tikhonov noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_gauss_elli model=tikhonov noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_uni_elli model=tikhonov noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_uni_elli model=tikhonov noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_uni_elli model=tikhonov noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform

/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_none_gauss_lodopa model=tikhonov noise_level=0.0 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_gauss_lodopa model=tikhonov noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_gauss_lodopa model=tikhonov noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_gauss_lodopa model=tikhonov noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_uni_lodopa model=tikhonov noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_uni_lodopa model=tikhonov noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_uni_lodopa model=tikhonov noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform



/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_none_gauss_lodopa model=filter noise_level=0.0 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_gauss_lodopa model=filter noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_gauss_lodopa model=filter noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_gauss_lodopa model=filter noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_uni_lodopa model=filter noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_uni_lodopa model=filter noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_uni_lodopa model=filter noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform

/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_none_gauss_elli model=filter noise_level=0.0 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_gauss_elli model=filter noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_gauss_elli model=filter noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_gauss_elli model=filter noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_uni_elli model=filter noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_uni_elli model=filter noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_uni_elli model=filter noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=20 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform



/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_none_gauss_lodopa model=svd noise_level=0.0 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_gauss_lodopa model=svd noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_gauss_lodopa model=svd noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_gauss_lodopa model=svd noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_uni_lodopa model=svd noise_level=0.005 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_uni_lodopa model=svd noise_level=0.015 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_uni_lodopa model=svd noise_level=0.03 trainval_dataset=lodopab test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform

/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_none_gauss_elli model=svd noise_level=0.0 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_gauss_elli model=svd noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_gauss_elli model=svd noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_gauss_elli model=svd noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_uni_elli model=svd noise_level=0.005 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_uni_elli model=svd noise_level=0.015 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_uni_elli model=svd noise_level=0.03 trainval_dataset=ellipses test_dataset=ellipses epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform


















/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_none_gauss_trans model=tikhonov noise_level=0.0 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_gauss_trans model=tikhonov noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_gauss_trans model=tikhonov noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_gauss_trans model=tikhonov noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_low_uni_trans model=tikhonov noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_med_uni_trans model=tikhonov noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=tik_high_uni_trans model=tikhonov noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=0 test_batch_size=1 test_batch_count=10 noise_type=uniform



/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_none_gauss_trans model=filter noise_level=0.0 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_gauss_trans model=filter noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_gauss_trans model=filter noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_gauss_trans model=filter noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_low_uni_trans model=filter noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_med_uni_trans model=filter noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=fft_high_uni_trans model=filter noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=10 mode=learned test_batch_size=1 test_batch_count=10 noise_type=uniform



/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_none_gauss_trans model=svd noise_level=0.0 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_gauss_trans model=svd noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_gauss_trans model=svd noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_gauss_trans model=svd noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=gaussian
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_low_uni_trans model=svd noise_level=0.005 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_med_uni_trans model=svd noise_level=0.015 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform
/home/alexander/anaconda3/envs/FSDLIP/bin/python /home/alexander/Projects/convergent-data-driven-regularizations-for-ct-reconstruction/main.py hydra.job.name=svd_high_uni_trans model=svd noise_level=0.03 trainval_dataset=ellipses test_dataset=lodopab epochs=1 mode=analytic test_batch_size=1 test_batch_count=10 noise_type=uniform