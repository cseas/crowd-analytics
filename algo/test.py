import multiprocessing.pool as mpool

def worker(task):
    # work on task
    print(task)     # substitute your migration code here.

# create a pool of 10 threads
pool = mpool.ThreadPool(10)
N = 100

for task in range(N):
    pool.apply_async(worker, args = (task, ))

pool.close()
pool.join()