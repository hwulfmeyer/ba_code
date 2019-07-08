#taskset --cpu-list 0,1,2 python3 main.py --filepr gp_len_eplexF2 --no-pgp --cmplx length --sel eplex --el_size 1 --n_jobs 3 &
#taskset --cpu-list 3 python3 main.py --filepr gp_len_tour --no-pgp --cmplx length --sel tournament --el_size 1 --n_jobs 1 &
#taskset --cpu-list 4 python3 main.py --filepr pgp_len_tour --pgp --cmplx length --sel tournament --el_size 0 --n_jobs 1 &
#taskset --cpu-list 5 python3 main.py --filepr pgp_kom_tour --pgp --cmplx kommenda --sel tournament --el_size 0 --n_jobs 1 &
#taskset --cpu-list 6,7,8 python3 main.py --filepr gp_len_eplexF1 --no-pgp --cmplx length --sel eplex --el_size 1 --n_jobs 3


#taskset --cpu-list 4,5 python3 main.py --filepr gp_len_eplexF5 --no-pgp --cmplx length --sel eplex --el_size 1 --n_jobs 2

#taskset --cpu-list 0,1,2 python3 main.py --filepr eplex --no-pgp --cmplx length --sel eplex --el_size 1 --n_jobs 3 &
#taskset --cpu-list 0 python3 main.py --filepr pgp_len --pgp --cmplx length --sel tournament --el_size 0 --n_jobs 1 &
taskset --cpu-list 1,2 python3 main.py --filepr pgp_kom --pgp --cmplx kommenda --sel tournament --el_size 0 --n_jobs 2
#taskset --cpu-list 0,1,2,3 python3 main.py --filepr stgp --no-pgp --cmplx length --sel tournament --el_size 1 --n_jobs 4