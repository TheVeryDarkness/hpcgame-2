// Copilot, thanks for your help!
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>
#include <mutex>

const int TREE = 1;  // 树木
const int FIRE = 2;  // 火焰
const int ASH = 3;   // 灰烬
const int EMPTY = 0; // 空地

// 事件类型
// 1: 天降惊雷
//    将 (x1, y1) 的所有树木变为火焰
// 2: 妙手回春
//    将 [x1, x2] * [y1, y2] 的所有灰烬变为树木
struct Event {
    int ts;         // 事件触发的时间步
    int type;       // 事件类型：1（天降惊雷），2（妙手回春）
    int x1, y1;     // 事件的坐标或区域范围
    int x2, y2;     // 仅用于“妙手回春”事件
};

constexpr size_t threads = 16;
constexpr size_t thread_dim = 4;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }
    const char *input_file = argv[1];
    const char *output_file = argv[2];

    // 读取输入文件
    // size: 森林的宽度和长度
    // event_count: 事件数量
    // time_steps: 时间步数量
    int size, event_count, time_steps;

    std::ifstream fin(input_file);
    fin >> size >> event_count >> time_steps;

    // 初始化 MPI
    MPI_Init(&argc, &argv);
    // 进程编号，进程总数
    // 本题中进程总数为 4，因此我们将二维的世界分为 4 个区域，每个区域的宽度和长度为 size / 2，方格数为 size * size / 4
    // 编号为 0 的进程负责处理第 0 行到第 size / 2 - 1 行，编号为 1 的进程负责处理第 size / 2 行到第 size - 1 行
    // 编号为 2 的进程负责处理第 0 列到第 size / 2 - 1 列，编号为 3 的进程负责处理第 size / 2 列到第 size - 1 列
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 计算当前进程负责处理的区域
    const int block_size = size / 2;
    const int x_start = (rank % 2) * block_size;
    const int y_start = (rank / 2) * block_size;
    const int x_end = x_start + block_size - 1;
    const int y_end = y_start + block_size - 1;

    // 初始化森林
    std::vector<int> forest(block_size * block_size, TREE);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (x_start <= i && i <= x_end && y_start <= j && j <= y_end) {
                fin >> forest[(i - x_start) * block_size + j - y_start];
            } else {
                int temp;
                fin >> temp;
            }
        }
    }

    // 读取事件
    // events: 事件列表，按时间步递增排序
    std::vector<Event> events(event_count);
    for (int i = 0; i < event_count; i++) {
        fin >> events[i].ts >> events[i].type >> events[i].x1 >> events[i].y1;
        if (events[i].type == 2) {
            fin >> events[i].x2 >> events[i].y2;
        }
    }
    fin.close();

    // 将要处理的事件编号
    int event_id = 0;

    // 模拟火灾
    for (int t = 0; t < time_steps; t++) {
        // 处理事件后的森林
        std::vector<int> new_forest = forest;

        // 处理事件
        if (event_id < event_count && events[event_id].ts == t) {
            const Event &event = events[event_id];
            if (event.type == 1) {
                // 天降惊雷
                if (x_start <= event.x1 && event.x1 <= x_end && y_start <= event.y1 && event.y1 <= y_end) {
                    new_forest[(event.x1 - x_start) * block_size + event.y1 - y_start] = FIRE;
                }
            } else if (event.type == 2) {
                // 妙手回春
                #pragma omp parallel for collapse(2) // 并行化，不会数据竞争
                for (int x = std::max(event.x1, x_start); x <= std::min(event.x2, x_end); x++) {
                    for (int y = std::max(event.y1, y_start); y <= std::min(event.y2, y_end); y++) {
                        auto &cell = new_forest[(x - x_start) * block_size + y - y_start];
                        if (cell == ASH) {
                            cell = TREE;
                        }
                        // new_forest[x - x_start][y - y_start] = TREE;
                    }
                }
            }
            event_id++;
        }

        // 扩散火焰后的森林
        std::vector<int> new_new_forest = forest;

        // 扩散火焰
        // 由于我们使用了二维数组，因此我们需要考虑边界情况
        // 每个时间步，我们将当前区域的火焰扩散到周围的区域（即上下左右四个方向）
        // 如果当前区域的边界上有火焰，且未超出世界边界的话，我们将火焰扩散到对应的相邻区域
        // 由于只有 4 个区域，因此最多只有 2 个方向需要扩散，分别是左右（x 轴方向）和上下（y 轴方向）

        std::vector<int> send_data[2];
        std::vector<int> recv_data[2];
        recv_data[0].resize(block_size, -1);
        recv_data[1].resize(block_size, -1);

        // 当可能访问到边界时，需要加锁，以免多个线程同时访问同一个位置
        std::mutex mtx;

        const int thread_block_size = block_size / thread_dim;

        #pragma omp parallel for
        for (int t = 0; t < threads; ++t){
            const int x_start = (t % thread_dim) * thread_block_size;
            const int y_start = (t / thread_dim) * thread_block_size;
            const int x_end = x_start + thread_block_size - 1;
            const int y_end = y_start + thread_block_size - 1;

            for (int x = x_start; x <= x_end; x++) {
                for (int y = y_start; y <= y_end; y++) {
                    if (new_forest[x * block_size + y] == FIRE) {
                        const bool need_lock = (x <= 1 || x >= block_size - 2 || y <= 1 || y >= block_size - 2);
                        if (need_lock) {
                            mtx.lock();
                        }
                        if (x == 0 || x == block_size - 1) {
                            send_data[0].push_back(y);
                        }
                        if (y == 0 || y == block_size - 1) {
                            send_data[1].push_back(x);
                        }
                        if (x > 0 && new_forest[(x - 1) * block_size + y] == TREE) {
                            new_new_forest[(x - 1) * block_size + y] = FIRE;
                        }
                        if (x < block_size - 1 && new_forest[(x + 1) * block_size + y] == TREE) {
                            new_new_forest[(x + 1) * block_size + y] = FIRE;
                        }
                        if (y > 0 && new_forest[x * block_size + (y - 1)] == TREE) {
                            new_new_forest[x * block_size + (y - 1)] = FIRE;
                        }
                        if (y < block_size - 1 && new_forest[x * block_size + (y + 1)] == TREE) {
                            new_new_forest[x * block_size + (y + 1)] = FIRE;
                        }
                        if (need_lock) {
                            mtx.unlock();
                        }
                    }
                }
            }
        }
        MPI_Sendrecv(
            send_data[0].data(), send_data[0].size(), MPI_INT, (rank + 1) % 2, t,
            recv_data[0].data(), recv_data[0].size(), MPI_INT, (rank + 1) % 2, t,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        MPI_Sendrecv(
            send_data[1].data(), send_data[1].size(), MPI_INT, (rank + 2) % 4, t,
            recv_data[1].data(), recv_data[1].size(), MPI_INT, (rank + 2) % 4, t,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        for (int i = 0; i < recv_data[0].size(); i++) {
            const int x = (rank % 2 == 0) ? block_size - 1 : 0;
            const int y = recv_data[0][i];
            if (y < 0 || y >= block_size) {
                break;
            }
            auto &cell = new_new_forest[x * block_size + y];
            if (cell == TREE) {
                cell = FIRE;
            }
        }
        for (int i = 0; i < recv_data[1].size(); i++) {
            const int y = (rank / 2 == 0) ? block_size - 1 : 0;
            const int x = recv_data[1][i];
            if (x < 0 || x >= block_size) {
                break;
            }
            auto &cell = new_new_forest[x * block_size + y];
            if (cell == TREE) {
                cell = FIRE;
            }
        }

        // forest = new_forest;
        std::swap(forest, new_new_forest);
    }

    // 收集结果
    if (rank == 0) {
        std::vector<int> result(size * size);
        // 将自己的区域复制到结果中
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                result[i * size + j] = forest[i * block_size + j];
            }
        }
        // 接收其他进程的区域
        for (int r = 1; r < world_size; r++) {
            std::vector<int> block(block_size * block_size);
            MPI_Recv(block.data(), block_size * block_size, MPI_INT, r, time_steps, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < block_size; i++) {
                const int x = (r % 2) * block_size + i;
                for (int j = 0; j < block_size; j++) {
                    const int y = (r / 2) * block_size + j;
                    result[x * size + y] = block[i * block_size + j];
                }
            }
        }

        // 输出结果
        std::ofstream fout(output_file);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                fout << result[i * size + j] << " ";
            }
            fout << std::endl;
        }
        fout.close();
    } else {
        MPI_Send(forest.data(), block_size * block_size, MPI_INT, 0, time_steps, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}