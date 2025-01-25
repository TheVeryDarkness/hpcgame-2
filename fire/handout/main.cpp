// Copilot, thanks for your help!
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <vector>

const int TREE = 1;
const int FIRE = 2;
const int ASH = 3;
const int EMPTY = 0;

struct Event {
    int ts;         // 事件触发的时间步
    int type;       // 事件类型：1（天降惊雷），2（妙手回春）
    int x1, y1;     // 事件的坐标或区域范围
    int x2, y2;     // 仅用于“妙手回春”事件
};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }
    const char *input_file = argv[1];
    const char *output_file = argv[2];

    // 读取输入文件
    int size, event_count, time_steps;

    std::ifstream fin(input_file);
    fin >> size >> event_count >> time_steps;
    std::vector<Event> events(event_count);
    for (int i = 0; i < event_count; i++) {
        fin >> events[i].ts >> events[i].type >> events[i].x1 >> events[i].y1;
        if (events[i].type == 2) {
            fin >> events[i].x2 >> events[i].y2;
        }
    }
    fin.close();

    // 初始化 MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 计算每个进程的工作量
    int block_size = size / world_size;
    int block_start = rank * block_size;
    int block_end = (rank == world_size - 1) ? size : block_start + block_size;

    // 计算每个进程的事件数量
    int event_start = 0;
    int event_end = event_count;
    for (int i = 0; i < event_count; i++) {
        if (events[i].y1 >= block_start && events[i].y1 < block_end) {
            event_start = i;
            break;
        }
    }

    for (int i = event_count - 1; i >= 0; i--) {
        if (events[i].y1 >= block_start && events[i].y1 < block_end) {
            event_end = i + 1;
            break;
        }
    }

    // 初始化森林
    std::vector<std::vector<int>> forest(block_size, std::vector<int>(size, TREE));

    // 模拟火灾
    for (int t = 0; t < time_steps; t++) {
        // 天降惊雷
        for (int i = event_start; i < event_end; i++) {
            if (events[i].ts == t && events[i].type == 1) {
                if (events[i].x1 >= block_start && events[i].x1 < block_end) {
                    forest[events[i].x1 - block_start][events[i].y1] = FIRE;
                }
            }
        }

        // 传播火灾
        std::vector<std::vector<int>> new_forest(block_size, std::vector<int>(size, TREE));
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < size; j++) {
                if (forest[i][j] == FIRE) {
                    new_forest[i][j] = ASH;
                    if (i > 0 && forest[i - 1][j] == TREE) {
                        new_forest[i - 1][j] = FIRE;
                    }
                    if (i < block_size - 1 && forest[i + 1][j] == TREE) {
                        new_forest[i + 1][j] = FIRE;
                    }
                    if (j > 0 && forest[i][j - 1] == TREE) {
                        new_forest[i][j - 1] = FIRE;
                    }
                    if (j < size - 1 && forest[i][j + 1] == TREE) {
                        new_forest[i][j + 1] = FIRE;
                    }
                }
            }
        }
        forest = new_forest;

        // 妙手回春
        for (int i = event_start; i < event_end; i++) {
            if (events[i].ts == t && events[i].type == 2) {
                if (events[i].x1 >= block_start && events[i].x1 < block_end) {
                    for (int x = events[i].x1 - events[i].x2; x <= events[i].x1 + events[i].x2; x++) {
                        for (int y = events[i].y1 - events[i].y2; y <= events[i].y1 + events[i].y2; y++) {
                            if (x >= 0 && x < block_size && y >= 0 && y < size) {
                                forest[x][y] = TREE;
                            }
                        }
                    }
                }
            }
        }
    }

    // 收集结果
    if (rank == 0) {
        std::vector<std::vector<int>> result(size, std::vector<int>(size));
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = forest[i][j];
            }
        }
        for (int r = 1; r < world_size; r++) {
            std::vector<std::vector<int>> block(block_size, std::vector<int>(size));
            MPI_Recv(&block[0][0], block_size * size, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < size; j++) {
                    result[r * block_size + i][j] = block[i][j];
                }
            }
        }

        // 输出结果
        std::ofstream fout(output_file);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                fout << result[i][j] << " ";
            }
            fout << std::endl;
        }
        fout.close();
    } else {
        MPI_Send(&forest[0][0], block_size * size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}