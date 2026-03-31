for i in 0.05 0.1 0.2 0.3 
do
    echo "Running attack evaluation with epsilon = $i"
    python test.py --dataset CIFAR10 --use_attack --attack_eps $i --seed 42 --eval_interval 1
    python test.py --dataset MNIST --use_attack --attack_eps $i --seed 42 --eval_interval 1
done;