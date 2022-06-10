mod render_layers;

use smallvec::SmallVec;

use bevy_math::Vec3A;
pub use render_layers::*;

use bevy_app::{CoreStage, Plugin};
use bevy_asset::{AssetEvent, Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_transform::components::GlobalTransform;
use bevy_transform::TransformSystem;
use bevy_utils::HashMap;

use crate::{
    camera::{Camera, CameraProjection, OrthographicProjection, PerspectiveProjection, Projection},
    mesh::Mesh,
    primitives::{Aabb, Frustum, Sphere},
};

/// User indication of whether an entity is visible
#[derive(Component, Clone, Reflect, Debug)]
#[reflect(Component, Default)]
pub struct Visibility {
    pub is_visible: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self { is_visible: true }
    }
}

/// Algorithmically-computed indication of whether an entity is visible and should be extracted for rendering
#[derive(Component, Clone, Reflect, Debug)]
#[reflect(Component)]
pub struct ComputedVisibility {
    pub is_visible: bool,
}

impl Default for ComputedVisibility {
    fn default() -> Self {
        Self { is_visible: true }
    }
}

/// Use this component to opt-out of built-in frustum culling for Mesh entities
#[derive(Component)]
pub struct NoFrustumCulling;

#[derive(Clone, Component, Default, Debug, Reflect)]
#[reflect(Component)]
pub struct VisibleEntities {
    #[reflect(ignore)]
    pub entities: Vec<Entity>,
}

impl VisibleEntities {
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Entity> {
        self.entities.iter()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

/// Tracks which [`Entities`](Entity) have which meshes for entities whose [`Aabb`]s are managed by
/// the [`calculate_bounds`] and [`update_bounds`] systems.
#[derive(Debug, Default, Clone)]
pub struct EntityMeshRelationships {
    entities_with_mesh: HashMap<Handle<Mesh>, SmallVec<[Entity; 1]>>,
    mesh_for_entity: HashMap<Entity, Handle<Mesh>>,
}

impl EntityMeshRelationships {
    /// Register the passed `entity` as having the passed `mesh_handle`.
    fn register(&mut self, entity: Entity, mesh_handle: &Handle<Mesh>) {
        self.entities_with_mesh
            .entry(mesh_handle.clone_weak())
            .or_default()
            .push(entity);
        self.mesh_for_entity
            .insert(entity, mesh_handle.clone_weak());
    }

    /// Deregisters the relationship between an `Entity` and `Mesh`. Used so [`update_bounds`] can
    /// track which relationships are still active so `Aabb`s are updated correctly.
    fn deregister(&mut self, entity: Entity) {
        if let Some(mesh) = self.mesh_for_entity.remove(&entity) {
            if let Some(entities) = self.entities_with_mesh.get_mut(&mesh) {
                if let Some(idx) = entities.iter().position(|&e| e == entity) {
                    entities.swap_remove(idx);
                }
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VisibilitySystems {
    CalculateBounds,
    UpdateBounds,
    UpdateOrthographicFrusta,
    UpdatePerspectiveFrusta,
    UpdateProjectionFrusta,
    CheckVisibility,
}

pub struct VisibilityPlugin;

impl Plugin for VisibilityPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        use VisibilitySystems::*;

        app.init_resource::<EntityMeshRelationships>()
            .add_system_to_stage(
                CoreStage::PostUpdate,
                calculate_bounds.label(CalculateBounds),
            )
            .add_system_to_stage(CoreStage::PostUpdate, update_bounds.label(UpdateBounds))
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_frusta::<OrthographicProjection>
                    .label(UpdateOrthographicFrusta)
                    .after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_frusta::<PerspectiveProjection>
                    .label(UpdatePerspectiveFrusta)
                    .after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_frusta::<Projection>
                    .label(UpdateProjectionFrusta)
                    .after(TransformSystem::TransformPropagate),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                check_visibility
                    .label(CheckVisibility)
                    .after(CalculateBounds)
                    .after(UpdateOrthographicFrusta)
                    .after(UpdatePerspectiveFrusta)
                    .after(UpdateProjectionFrusta)
                    .after(TransformSystem::TransformPropagate),
            );
    }
}

/// Calculates [`Aabb`]s for [`Entities`](Entity) with [`Mesh`]es. To opt out of bound calculation
/// for an `Entity`, give it the [`NoFrustumCulling`] component.
pub fn calculate_bounds(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    without_aabb: Query<(Entity, &Handle<Mesh>), (Without<Aabb>, Without<NoFrustumCulling>)>,
    mut entity_mesh_rel: ResMut<EntityMeshRelationships>,
) {
    for (entity, mesh_handle) in without_aabb.iter() {
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(aabb) = mesh.compute_aabb() {
                entity_mesh_rel.register(entity, mesh_handle);
                commands.entity(entity).insert(aabb);
            }
        }
    }
}

/// Updates [`Aabb`]s for [`Entities`](Entity) with [`Mesh`]es. This includes `Entities` that have
/// been assigned new `Mesh`es as well as `Entities` whose `Mesh` has been directly mutated.
///
/// To opt out of bound calculation for an `Entity`, give it the [`NoFrustumCulling`] component.
///
/// **Note** This system needs to remove entities from their collection in
/// [`EntityMeshRelationships`] whenever a mesh handle is reassigned or an entity's mesh handle is
/// removed. This may impact performance if meshes with many entities are frequently
/// reassigned/removed.
pub fn update_bounds(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    mut mesh_reassigned: Query<
        (Entity, &Handle<Mesh>, &mut Aabb),
        (Changed<Handle<Mesh>>, Without<NoFrustumCulling>),
    >,
    mut entity_mesh_rel: ResMut<EntityMeshRelationships>,
    mut mesh_events: EventReader<AssetEvent<Mesh>>,
    entities_lost_mesh: RemovedComponents<Handle<Mesh>>,
) {
    for entity in entities_lost_mesh.iter() {
        entity_mesh_rel.deregister(entity);
    }

    for (entity, mesh_handle, mut aabb) in mesh_reassigned.iter_mut() {
        entity_mesh_rel.deregister(entity);
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(new_aabb) = mesh.compute_aabb() {
                entity_mesh_rel.register(entity, mesh_handle);
                *aabb = new_aabb;
            }
        }
    }

    let to_update = |event: &AssetEvent<Mesh>| {
        let handle = match event {
            AssetEvent::Modified { handle } => handle,
            _ => return None,
        };
        let mesh = meshes.get(handle)?;
        let entities_with_handle = entity_mesh_rel.entities_with_mesh.get(handle)?;
        let aabb = mesh.compute_aabb()?;
        Some((aabb, entities_with_handle))
    };
    for (aabb, entities_with_handle) in mesh_events.iter().filter_map(to_update) {
        for entity in entities_with_handle {
            commands.entity(*entity).insert(aabb.clone());
        }
    }
}

pub fn update_frusta<T: Component + CameraProjection + Send + Sync + 'static>(
    mut views: Query<(&GlobalTransform, &T, &mut Frustum)>,
) {
    for (transform, projection, mut frustum) in views.iter_mut() {
        let view_projection =
            projection.get_projection_matrix() * transform.compute_matrix().inverse();
        *frustum = Frustum::from_view_projection(
            &view_projection,
            &transform.translation,
            &transform.back(),
            projection.far(),
        );
    }
}

pub fn check_visibility(
    mut view_query: Query<(&mut VisibleEntities, &Frustum, Option<&RenderLayers>), With<Camera>>,
    mut visible_entity_query: ParamSet<(
        Query<&mut ComputedVisibility>,
        Query<(
            Entity,
            &Visibility,
            &mut ComputedVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&NoFrustumCulling>,
            Option<&GlobalTransform>,
        )>,
    )>,
) {
    // Reset the computed visibility to false
    for mut computed_visibility in visible_entity_query.p0().iter_mut() {
        computed_visibility.is_visible = false;
    }

    for (mut visible_entities, frustum, maybe_view_mask) in view_query.iter_mut() {
        visible_entities.entities.clear();
        let view_mask = maybe_view_mask.copied().unwrap_or_default();

        for (
            entity,
            visibility,
            mut computed_visibility,
            maybe_entity_mask,
            maybe_aabb,
            maybe_no_frustum_culling,
            maybe_transform,
        ) in visible_entity_query.p1().iter_mut()
        {
            if !visibility.is_visible {
                continue;
            }

            let entity_mask = maybe_entity_mask.copied().unwrap_or_default();
            if !view_mask.intersects(&entity_mask) {
                continue;
            }

            // If we have an aabb and transform, do frustum culling
            if let (Some(model_aabb), None, Some(transform)) =
                (maybe_aabb, maybe_no_frustum_culling, maybe_transform)
            {
                let model = transform.compute_matrix();
                let model_sphere = Sphere {
                    center: model.transform_point3a(model_aabb.center),
                    radius: (Vec3A::from(transform.scale) * model_aabb.half_extents).length(),
                };
                // Do quick sphere-based frustum culling
                if !frustum.intersects_sphere(&model_sphere, false) {
                    continue;
                }
                // If we have an aabb, do aabb-based frustum culling
                if !frustum.intersects_obb(model_aabb, &model, false) {
                    continue;
                }
            }

            computed_visibility.is_visible = true;
            visible_entities.entities.push(entity);
        }

        // TODO: check for big changes in visible entities len() vs capacity() (ex: 2x) and resize
        // to prevent holding unneeded memory
    }
}
