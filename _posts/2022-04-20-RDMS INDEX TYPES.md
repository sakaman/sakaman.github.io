---
layout: post
title: 深入索引一点点
categories:
  - General
  - Database
---

对 CS 554 中索引类型的一些总结

# 索引是什么？

> On the most fundamental level, a database needs to do things: when you give it some data, it should store the data,and
> when you ask it again later, it should give the data back to you. -- Designing Data-Intensive Applications

在日均增长亿量数据的互联网时代，网络服务每时每刻都在与这些数据打交道，搜索、分析、转换等。时间就是成本，如何高效处理这些数据将成为难题。为减少查
询数据库的开销，计算机科学家设计了各种算法来处理不同的场景，同时引进一种数据结构：**索引(index)**，来支持高效查询数据库中特定的值。索引大致的
工作方式即是保存一些额外的元数据作为标记，帮助查找所需数据。如果需要在同一份数据中用不同的方式搜索，可能需要不同的索引，建在数据的不同部分上。索
引是从主数据衍生的附加结构。许多数据库允许添加与删除索引，这不会影响数据的内容，只影响查询的性能。维护额外的结构会产生开销，特别是在写入时。任何
类型的索引通常都会减慢写入速度，因为每次写入数据时都需要更新索引。因此，存储系统中重要的一个权衡：精心选择的索引加快了读查询的速度，但是每个索引
都会拖慢写入速度。所以，数据库默认不会索引所有的内容，而需要使用者通过对应用查询模式的了解来手动选择索引，为应用带来最大收益，同时又不会引入超出
必要开销的索引。

## 优缺点

Pros:

- 提高查询性能，高效读取数据

Cons:

- 占用磁盘与内存空间
- 维护开销，减慢写入性能（插入、更新、删除）

# 索引有哪些？

索引大致分为两类：

- 有序索引（Sorted index）
  - 搜索 keys 有序存储在索引文件中
  - 使用二分查找(binary search)快速搜索
- 哈希索引（Hashing index）
  - 搜索 keys 存储在哈希桶中
  - 使用哈希函数(hash function)快速搜索

其中，哈希索引（Hash index）是使用最广泛的一种索引，而有序索引的代表，则为 MySQL 中的 Primary index(an ordered index whose search
key is also the sort key used for sequential file)和 Secondary index(an ordered index whose search key is not the sort
key used for the sequential file)，还有其他形式的索引，比如 LevelDB 的 SSTable，ElasticSearch 中的倒排索引（Reverse index），
BitMap，etc,.

## 哈希索引 - Hashing Table

> Hash Function: a function that maps a search key to an index between[0 .. B-1] (B = the size of the hash table)

通常应用于键值数据（key-value Data）。在大多数编程语言中都能找到类似字典（dictionary）类型数据结构，通常用散列映射（hash map）或者
哈希表（hash table）实现，具有 O(1)极高效的查找效率。

### **数据存储形式**

![How to use hashing as a indexing technique to find records stored on disk](Hash%20store.png)

- 一个 bucket 对应一个 disk block
- 一个 bucket 包含 n 个（key，ptr）对

### **存储与查找**

![Store & Find hash tables on disk](Hash%20query.png)
**Steps**

1. Compute the hash value h(W)
2. Read the disk block(s) of bucket h(W) into memory
3. Search (a linear search algorithm is sufficient because memory search is fast) the bucket h(W) for: **W, RecordPtr(W)**
4. Use RecordPtr(W) to access W on disk

### Hashing animation

![Hashing](Hashing.gif)

### 扩容

当哈希表插入许多值时，将会产生许多溢出块（哈希冲突），此时将会产生更多的磁盘读取操作，降低性能。通过增加哈希表的容量是一种普遍有效的解决方法，但
是成本很高，通常需要重新映射所有 keys 值至新哈希表。因而提出两种动态哈希方法，当哈希表的大小改变时，都只需要重新映射小部分已存在的 keys。

- 可扩展哈希（[Extensible Hashing](https://en.wikipedia.org/wiki/Extendible_hashing)）：能完全消除溢出数据的影响，但是哈希容量呈指数增长
- 线性哈希（[Linear Hashing](<https://en.wikipedia.org/wiki/Linear_hashing#:~:text=Linear%20hashing%20(LH)%20is%20a,%2DYates%20and%20Soza%2DPollman.>)）：不能完全消除溢出数据的影响，但是哈希容量呈线性增长

### 优缺点

Pros:

- 单数据行 O(1)操作性能

Cons:

- 对于范围查询和排序无法很好支持，最终可能导致全表扫描
- 需要足够内存放入散列表（在大量数据时，对于磁盘哈希映射，需要大量的随机访问 I/O，且无法高效处理 Resizing 和 Collision）
- 无法支持模糊搜索

## 有序索引 - B-Tree and B+-tree

根据公开资料，读取磁盘数据块通常在[0.01 秒内](http://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/3-index/2-disks/structure3.html)，当计算数据库操作的延迟时，主要考虑磁盘随机访问次数。对于单一有序索引，使用二分查找算法，最糟糕的耗时为 O(logn)；如果是多层有序索引，能够极大降低磁盘 I/O 次数，这时，索引文件的层数将成为关键，这取决于目标 key 在索引中的位置（ground level）；如果 keys 越多，索引可能需要更多的层数。此时，B+ tree 能够根据索引数量动态判断索引的层数。
![disk access time](disk%20access%20time.png)

### 索引层数与访问次数

![access time](B+tree.png)

- 图示三次磁盘访问即可查找到数据地址
- 因此，为降低磁盘 I/O 次数，必须降低索引层数

### B+ Tree 的定义

![B+ tree](B+tree%20definition.png)

- 动态多层级(3 层可储存约 4 千万级数据)
- 平衡树（每个叶子结点到根结点距离相同）
- 每个节点含有多个数据地址（每个节点占据一个磁盘块，为页的整数倍（4KB），一个 block 可储存 340【4n+8(n+1) <= 4096】个 keys，四层可保存 53T 数据）
- 叶子节点存储所有数据「index file」（密集索引）
- 叶子节点通过指针按顺序连接，即包含一个指向下一邻近叶节点的指针。

### 叶子节点的结构

![B+ tree leaf](assets/images/2022-04-20-INDEX/paint-self/leaf%20node.png)

- 每个叶子节点(leaf node)存储在一个磁盘块(disk block)
- 索引键即为想要快速查找的值
- 叶子节点的指针（除最后的指针）为关联索引键记录数据库数据（database record）的地址
- 叶子节点从左至右链接

### 查找 B-tree

- 起始在根结点使用**线性搜索**查询下一个节点
  ![Lookup B-tree](B%20tree%20query.png)
- 在**内部节点**中重复该步骤
- 当到达**叶节点**时，线性搜索目标 key
  ![Lookup leaf node](leaf%20node%20query.png)

### 插入新值

![B+ tree](B+%20tree.gif)

- 若叶节点有空余空间，查找对应叶节点，位移 keys 并插入目标 key
- 若叶节点空间已满，查找对应叶节点，插入目标 key，并等半分裂为两个节点，在父节点插入两个新叶节点中的中值
- 若叶节点与父节点空间已满，查找对应叶节点，插入目标 key，等半分裂；父节点插入两个新叶节点的中值，分裂为两个节点；将中值移入上级父节点，重复此步骤
  B+树将数据库分解成固定大小的块或者页面，传统上大小为 4KB，并且一次只能读取或写入一个页面。这种设计更接近于底层硬件，因为磁盘也被划分为固定大
  小的块。通常大多数数据库可以放入一个三到四层的 B+树，极大降低对页面的查找次数，从而大幅减少磁盘随机访问 I/O 次数。

而对于 B 树，能够在非叶节点存储数据，可导致查询**连续**数据时产生更多的随机 I/O，而 B+树的所有叶节点通过指针连接，能够减少顺序遍历时产生的额外随机 I/O。
![B tree](B%20tree.gif)

### 优化

- 当然可以同时构建 B+树和哈希表的混合存储结构，来达到极致的性能，但也会带来更高的复杂度，在维护数据结构时，会导致更新和删除时需要操作更多份数据。
- 由于索引对只适用于直接访问（从左到右扫描匹配），因此通常能压缩索引；经过前缀压缩，可以有效提高 B+树扇出，降低树高，提高查找效率。
- 对于大量数据，重复迭代插入 B+树将会非常慢，此时，批量插入将会大幅提高效率。首先将所有数据进行排序，然后插入索引至叶节点首页或者尾页，最后重新
  分布节点。有效降低磁盘随机访问次数，优化并发控制。
- 所有叶节点自然排序，能够有效提高范围查找和邻近查找。

## 多维索引

无论是哈希索引还是有序索引都是一维索引，但是针对于类似**地理空间信息**等数据，亦或对于临近点、包含关系的查找等，一维索引有点难以应付，此时，便
引入**多维索引**：建立在多维数据上的一种索引，支持有效多维查询，应用于**部分匹配查找(Partial Match)、范围查找(Range)、最近临近点(Nearest neighbor)
查找、位置查找(Where-am-I, Ponit)**等。
![Multi-dimensional indexes applications](Multi-dimensional%20queries.png)

### 多维索引结构

- Table-based
  - Grid index files
  - Partitioned Hashing
- Tree-like(Tree based)
  - Multiple key indexes
  - kd-tree
  - Quad-tree
  - R-tree

### Grid Index file

> 构造成二维结构的索引

![Grid Index](assets/images/2022-04-20-INDEX/paint-self/grid%20index.png)

- 网格索引文件存储 m、n 大小的网格，存储网格桶(Buckets)，包含 m**n 或(m+1)**(n+1)个块指针
- 很容易扩展至高维索引
- 网格线有两种含义：
  - 网格线表示独立的点，即网格块存储对应键值（坐标值指向网格块）
  - 网格线表示范围，网格块存储特范围键值（坐标值指向网格线）

#### 查找

- 查找横向索引（x）的位置
- 查找纵向索引（y）的位置
- 查找数据块指针偏移量
  - offset = row index \* (column index) + column index
- 在数据块中查找对应数据

#### 插入

1.  定位待插入位置「bucket」
2.  如果有空间则插入数据
3.  若无，链接溢出块或者拆分 bucket

#### 多维查找情景

> 假设能够全部存储在内存中

- 无法表示对象，故不能支持 where-am-I queries
- 数据分布不均匀时，将产生许多空白空间
- 需要良好的算法支撑空间划分
  ![Grid Lookup](Grid%20query.png)
  ![Partitioned Hashing occupancy](Partitioned%20hashing%20occupancy%20rate.png)

### Partitioned Hashing

通常使用的 Hashing 无法解决组合值「多维数据」问题；针对 n 个组合 key，Partitioned Hashing 使用 n 个哈希函数，每个函数对应一个子 key，
哈希值即是这些单独哈希函数值的**连接组合**。
![Partitioned Hashing](assets/images/2022-04-20-INDEX/paint-self/partitioned%20hashing.png)

#### **多维查找情景**

- Partial Match queries
  匹配部分子 key，通常被用于减少搜索空间
- Range queries
  通常不适用，因为哈希函数无法保存值的邻近性
- Nearest neighbor queries
  不适用，由于哈希函数值是随机的，无法保存数据之间的真实距离
- Where-am-I queries
  不适用，哈希函数不提供任何距离信息
  但是 Partitioned Hashing 能提供良好的分布率，相比 Grid index 使用更少的空间，可结合其他索引结构使用。

### Multiple-key index

> 多层级索引在不同维度使用不同类型（B-tree、Hashing）的索引

- Partial Match queries
  搜索值在第一层索引时，是非常有效的。
- Range queries
  查找部分或者全部有确定范围的数据。
- Nearest neighbor queries
  扩展使用范围搜索「Range queries」查找最邻近点。
- Where-am-I
  不适用
  ![Multiple-index queries](Multi-dimensional%20queries.png)

### kd-tree

- kd-tree 是二叉搜索树（BST）的一种
- 在不同层级使用的搜索 key 属于不同的维度
- 不同层级上的维度会包围起来
  ![kd-tree](assets/images/2022-04-20-INDEX/paint-self/kd-tree.png)

#### 查找

类似 BST

#### 插入

类似 BST

- Partial Match queries
  对于给定搜索值的维度，获取其中一个子树的值；反之，获取所有子树的值。
- Range queries
  根据搜索范围获取对应子树的值
- Nearest neighbor queries
  不适用
- Where-am-I
  不适用
  ![kd tree](kd-tree%20query.png)
  ![kd-tree](kd-tree.gif)

### Quad-tree

一种每个维度对半划分的索引结构
![Quad-tree](assets/images/2022-04-20-INDEX/paint-self/Quad-tree.png)
简化的 kd-tree，多维搜索类似 kd-tree
![quad2kd](quad-tree2kd-tree.png)

### R-tree

从 B-tree 衍生的一种使用边界盒（bounding boxes）作为搜索 key 的索引结构。

#### 查找

1.  从根节点开始
2.  查找包含 key 的区域节点
3.  若无，则不存在，查找结束
4.  反之，递归查找所有访俄条件的区域子节点
5.  到达叶节点时，查找到数据的位置

#### 插入

1.  从根节点开始，尝试查找 key 适合的区域
2.  若存在，往下重复查找；
3.  若不存在，需要扩展已存在的区域
    1.  尽量小的扩展
4.  当到达叶节点，插入 key
5.  若无空间，拆分区域节点
    1.  类似 B-tree

![R-tree](assets/images/2022-04-20-INDEX/paint-self/r-tree.png)

#### 多维查找

非常适用于 Partial Match queries, Range queries, Nearest neighbor queries, Where-am-I queries 四种情景，是多维索引一种高度理想结构。

# 索引怎么使用？

根据实际需求，选取不同的索引，达到空间与时间最优。通过了解索引的数据结构及算法，深入理解索引的优缺点及优化方式，比如最左前缀原理、索引长度、排序等。

# 引用参考

- https://github.com/Vonng/ddia/blob/master/en-us/ch3.md
- https://en.wikipedia.org/wiki/Database_index
- https://en.wikipedia.org/wiki/Hash_table
- https://en.wikipedia.org/wiki/Search_engine_indexing
- https://en.wikipedia.org/wiki/B%2B_tree
- https://draveness.me/whys-the-design-mysql-b-plus-tree/
- http://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/syl.html#CURRENT
- https://web.cs.ucdavis.edu/~green/courses/ecs165b-s10/Lecture6.pdf
- https://www.ibm.com/docs/en/db2woc?topic=objects-indexes
- https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf
- http://mlwiki.org/index.php/Indexing_(databases)
- https://github.com/davidmoten/rtree
- https://github.com/myui/btree4j
- https://github.com/linpc2013/KDTree
