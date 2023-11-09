from multiprocessing import Pipe, Process

def merge_sort(data, conn = None, parallel = False):
    if len(data) <= 1:
        if conn:
            conn.send(data)
            conn.close()
        return data
    else:
        split = len(data) // 2

        if parallel: # Parallel merge sort
            # Create pipes for inter-process communication
            left_conn, left_conn_child = Pipe()
            right_conn, right_conn_child = Pipe()

            # Create processes for sorting each half of the list
            left_process = Process(target = merge_sort, args = (data[:split], left_conn_child))
            right_process = Process(target = merge_sort, args = (data[split:], right_conn_child))

            # Start the processes
            left_process.start()
            right_process.start()

            # Get sorted sub-lists from the child processes
            left_sorted = left_conn.recv()
            right_sorted = right_conn.recv()

            # Wait for processes to finish
            left_process.join()
            right_process.join()
        else: # Non-parallel merge sort
            left_sorted = merge_sort(data[:split])
            right_sorted = merge_sort(data[split:])

        # Merge sorted sub-lists
        result = []
        left, right = iter(left_sorted), iter(right_sorted)
        left_top, right_top = next(left, None), next(right, None)

        while left_top is not None or right_top is not None:
            if right_top is None or (left_top is not None and left_top < right_top):
                result.append(left_top)
                left_top = next(left, None)
            else:
                result.append(right_top)
                right_top = next(right, None)
        
        if conn:
            conn.send(result)
            conn.close()
            
        return result