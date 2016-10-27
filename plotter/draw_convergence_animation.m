function [  ] = draw_convergence_animation(problem, w_history)

figure;

    N = 50;
    edge = 1;
    x_min = 0.2;
    x_max = 0.6;    
    y_min = -0;
    y_max = 3;    
    
    alg_num = length(w_history);
    
    
    [w_min, f_min] = draw_3D_surface(problem.cost, N, x_min, x_max, y_min, y_max); hold on
    view(150,50);
    
    len_array = zeros(alg_num,1);
    for i=1:alg_num
        len_array(i) = size(w_history{i},2);
    end
    len_max = max(len_array);
    
    c_style = {'b','r','c'};    
    m_style = {'o','o','o'};
    l_style = {'-','-','-'};  
    
    W_prev = cell(alg_num,1);
    
    for i=1:len_max
        
        x_max = -Inf;
        x_min = Inf;  
        y_max = -Inf;
        y_min = Inf;           

        if i > 1
            fprintf('epoch: %d', i);        
            for j=1:alg_num
                w = w_history{j};
                w_prev = W_prev{j};
                if i <= len_array(j)
                    z = problem.cost(w(:,i));
                    %plot3(w(1,i), w(2,i), z, style1{j},'MarkerFaceColor',style{j}); hold on   
                    %line([.3 .7],[.4 .9],[1 1],'Marker','.','LineStyle','-'); hold on
                    %plot3(0,0,0, w(1,i), w(2,i), z, 'r-'); hold on  
                    line([w_prev(1) w(1,i)],[w_prev(2) w(2,i)],[w_prev(3) z],'Marker', m_style{j}, 'LineStyle',l_style{j}, 'Color', c_style{j}, 'LineWidth',2); hold on
                    
                    W_prev{j} = [w(1,i); w(2,i); z ];
                    
                    fprintf('[%d] (%.2f,%.2f,%.2f)   ', j, w(1,i), w(2,i),z );   
                    
                    if x_max <  w(1,i)
                        x_max = w(1,i);
                    end
                    if y_max <  w(2,i)
                        y_max = w(2,i);
                    end  
                    if x_min >  w(1,i)
                        x_min = w(1,i);
                    end      
                    if y_min >  w(2,i)
                        y_min = w(2,i);
                    end                      
                end
                
            %xlim([x_min-1, x_max+1]);
            %ylim([y_min-1, y_max+1]);  
            %draw_3D_surface(problem.cost, N, x_min-1, x_max+1, y_min-1, y_max+1); hold on
                                
            end
            fprintf('\n');
        else
            for j=1:alg_num
                w = w_history{j};
                z = problem.cost(w(:,i));
                w_tmp(1) = w(1,1);
                w_tmp(2) = w(2,1);  
                W_prev{j} = [w(1,1); w(2,1); z];
            end
        end
        
   


        pause(0.5)
    end
    hold off    
end

