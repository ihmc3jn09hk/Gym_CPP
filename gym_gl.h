#include <vector>

class Gym_Renderer_CartPoleContinuous
{
public:
	explicit Gym_Renderer_CartPoleContinuous(const int res_x = 128, const int res_y = 128);
	~Gym_Renderer_CartPoleContinuous();
	
	std::pair<int,int> render_state(std::vector<double> pos,
									 std::vector<double> ang,
									 std::vector<unsigned int>& data);
private:
	int m_iWidth;
	int m_iHeight;
};